@preconcurrency import CoreML
import Foundation
import Metal
import os

private let coreMLLog = OSLog(subsystem: "com.siliconpedia.LingBotDepth", category: .pointsOfInterest)

public enum CoreMLPlacement: String, CaseIterable, Sendable {
    case all, cpuAndNeuralEngine, cpuAndGPU, cpuOnly

    var computeUnits: MLComputeUnits {
        switch self {
        case .all: .all
        case .cpuAndNeuralEngine: .cpuAndNeuralEngine
        case .cpuAndGPU: .cpuAndGPU
        case .cpuOnly: .cpuOnly
        }
    }
}

public struct CoreMLOptimizationOptions: Sendable, Equatable {
    public let fastPrediction: Bool
    public let lowPrecisionGPUAccumulation: Bool

    public init(fastPrediction: Bool = false, lowPrecisionGPUAccumulation: Bool = false) {
        self.fastPrediction = fastPrediction
        self.lowPrecisionGPUAccumulation = lowPrecisionGPUAccumulation
    }

    public func configuration(for placement: CoreMLPlacement) -> MLModelConfiguration {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = placement.computeUnits
        configuration.optimizationHints.reshapeFrequency = .infrequent
        configuration.optimizationHints.specializationStrategy = fastPrediction ? .fastPrediction : .default
        configuration.allowLowPrecisionAccumulationOnGPU = lowPrecisionGPUAccumulation
        return configuration
    }
}

public actor CoreMLBackend: DepthEstimating {
    public nonisolated let profile: DepthProfile
    public nonisolated let kind = DepthBackendKind.coreML
    private let modelURL: URL
    private let placement: CoreMLPlacement
    private let optimization: CoreMLOptimizationOptions
    private var model: MLModel?
    /// Allowed compute units reported by the loaded model, not observed hardware execution.
    public var loadedComputeUnitsRawValue: Int? { model?.configuration.computeUnits.rawValue }
    private let inputNames: (image: String, depth: String, validity: String)
    private let outputNames: (depth: String, mask: String?)
    private var outputBackings: [OutputBackingSlot] = []

    public init(
        modelURL: URL,
        profile: DepthProfile,
        placement: CoreMLPlacement = .all,
        optimization: CoreMLOptimizationOptions = .init(),
        inputNames: (String, String, String) = ("image", "depth_m", "validity"),
        outputNames: (String, String?) = ("select_1", "sigmoid")
    ) {
        self.modelURL = modelURL
        self.profile = profile
        self.placement = placement
        self.optimization = optimization
        self.inputNames = inputNames
        self.outputNames = outputNames
    }

    public func load() async throws {
        guard model == nil else { return }
        os_signpost(.begin, log: coreMLLog, name: "ModelLoad")
        defer { os_signpost(.end, log: coreMLLog, name: "ModelLoad") }
        let configuration = optimization.configuration(for: placement)
        let url = modelURL.pathExtension == "mlmodelc" ? modelURL : try await MLModel.compileModel(at: modelURL)
        let loaded = try await MLModel.load(contentsOf: url, configuration: configuration)
        let shape = profile.shape
        let inputs = loaded.modelDescription.inputDescriptionsByName
        let expected = [inputNames.image: [1, 3, shape.height, shape.width],
                        inputNames.depth: [1, 1, shape.height, shape.width],
                        inputNames.validity: [1, 1, shape.height, shape.width]]
        guard Set(inputs.keys) == Set(expected.keys) else { throw DepthRuntimeError.unsupportedProfile(profile) }
        for (name, dimensions) in expected {
            guard let constraint = inputs[name]?.multiArrayConstraint, constraint.dataType == .float32,
                  constraint.shape.map(\.intValue) == dimensions else { throw DepthRuntimeError.unsupportedProfile(profile) }
        }
        let outputs = loaded.modelDescription.outputDescriptionsByName
        let expectedOutputs = [outputNames.depth, outputNames.mask].compactMap { $0 }
        let constraints = expectedOutputs.compactMap { outputs[$0]?.multiArrayConstraint }
        guard outputNames.depth != outputNames.mask,
              Set(outputs.keys) == Set(expectedOutputs), constraints.count == expectedOutputs.count,
              constraints.allSatisfy({ [.float16, .float32].contains($0.dataType) && $0.shape.map(\.intValue) == [1, shape.height, shape.width] }) else {
            throw DepthRuntimeError.unsupportedProfile(profile)
        }
        if constraints.allSatisfy({ $0.dataType == .float16 }), let device = MTLCreateSystemDefaultDevice() {
            outputBackings = try (0..<2).map { _ in
                let (depth, depthBuffer) = try makeSharedMultiArray(device: device, shape: [1, shape.height, shape.width], dataType: .float16)
                let mask = try outputNames.mask.map { _ in
                    try makeSharedMultiArray(device: device, shape: [1, shape.height, shape.width], dataType: .float16)
                }
                return OutputBackingSlot(depth: depth, mask: mask?.0, depthBuffer: depthBuffer, maskBuffer: mask?.1)
            }
        }
        model = loaded // Publish only after contract checks and backing allocation succeed.
    }

    public func prewarm() async throws {
        let shape = profile.shape
        let image = try MLMultiArray(shape: [1, 3, NSNumber(value: shape.height), NSNumber(value: shape.width)], dataType: .float32)
        let depth = try MLMultiArray(shape: [1, 1, NSNumber(value: shape.height), NSNumber(value: shape.width)], dataType: .float32)
        let validity = try MLMultiArray(shape: depth.shape, dataType: .float32)
        _ = try await predict(ModelTensors(image: image, depthMeters: depth, validity: validity, timestamp: 0))
    }

    public func predict(_ input: ModelTensors) async throws -> DepthResult {
        guard let model else { throw DepthRuntimeError.modelNotLoaded }
        os_signpost(.begin, log: coreMLLog, name: "ModelPrediction")
        defer { os_signpost(.end, log: coreMLLog, name: "ModelPrediction") }
        let features = try MLDictionaryFeatureProvider(dictionary: [
            inputNames.image: MLFeatureValue(multiArray: input.image),
            inputNames.depth: MLFeatureValue(multiArray: input.depthMeters),
            inputNames.validity: MLFeatureValue(multiArray: input.validity),
        ])
        let clock = ContinuousClock()
        let options = MLPredictionOptions()
        var reserved: (OutputBackingSlot, OutputBackingLease)?
        while !outputBackings.isEmpty {
            try Task.checkCancellation()
            guard self.model === model else { throw DepthRuntimeError.modelNotLoaded }
            for index in outputBackings.indices {
                if let lease = outputBackings[index].reserve() {
                    reserved = (outputBackings[index], lease)
                    break
                }
            }
            if reserved != nil { break }
            // ponytail: bounded two-slot pool; replace 1ms async polling with release notification if wait overhead matters.
            try await Task.sleep(for: .milliseconds(1))
        }
        guard self.model === model else { throw DepthRuntimeError.modelNotLoaded }
        let backing = reserved?.0
        if let backing {
            var arrays = [outputNames.depth: backing.depth]
            if let name = outputNames.mask, let mask = backing.mask { arrays[name] = mask }
            options.outputBackings = arrays
        }
        let started = clock.now // Slot waiting belongs to wrapper time, not model prediction time.
        let output = try await model.prediction(from: features, options: options)
        let latency = started.duration(to: clock.now)
        try Task.checkCancellation()
        guard self.model === model else { throw DepthRuntimeError.modelNotLoaded }
        guard let depth = output.featureValue(for: outputNames.depth)?.multiArrayValue else {
            throw DepthRuntimeError.missingOutput(outputNames.depth)
        }
        let mask = try outputNames.mask.map { name in
            guard let value = output.featureValue(for: name)?.multiArrayValue else {
                throw DepthRuntimeError.missingOutput(name)
            }
            return value
        }
        return DepthResult(
            depth: try reserved.map { try $0.1.retain(in: depth) } ?? depth,
            maskProbability: try mask.map { value in try reserved.map { try $0.1.retain(in: value) } ?? value },
            profile: profile,
            backend: kind,
            latency: latency,
            timestamp: input.timestamp,
            depthBuffer: backing.flatMap { Self.acceptedOutputBuffer(depth, backing: $0.depth, buffer: $0.depthBuffer) },
            maskBuffer: backing.flatMap { slot in
                guard let mask, let array = slot.mask, let buffer = slot.maskBuffer else { return nil }
                return Self.acceptedOutputBuffer(mask, backing: array, buffer: buffer)
            }
        )
    }

    // Pointer identity alone does not establish the layout consumed by Metal.
    nonisolated static func acceptedOutputBuffer(_ output: MLMultiArray, backing: MLMultiArray, buffer: MTLBuffer) -> MTLBuffer? {
        guard output.dataPointer == backing.dataPointer,
              output.dataType == backing.dataType,
              output.shape == backing.shape,
              output.strides == backing.strides else { return nil }
        return buffer
    }

    public func unload() {
        os_signpost(.event, log: coreMLLog, name: "ModelUnload")
        model = nil
        outputBackings.removeAll()
    }
}

// Actor-owned slots hold only a weak reservation; output array aliases own it.
struct OutputBackingSlot {
    let depth: MLMultiArray
    let mask: MLMultiArray?
    let depthBuffer: MTLBuffer
    let maskBuffer: MTLBuffer?
    private weak var lease: OutputBackingLease?

    init(depth: MLMultiArray, mask: MLMultiArray?, depthBuffer: MTLBuffer, maskBuffer: MTLBuffer?) {
        self.depth = depth
        self.mask = mask
        self.depthBuffer = depthBuffer
        self.maskBuffer = maskBuffer
    }

    mutating func reserve() -> OutputBackingLease? {
        guard lease == nil else { return nil }
        let reservation = OutputBackingLease()
        lease = reservation
        return reservation
    }
}

final class OutputBackingLease: Sendable {
    func retain(in output: MLMultiArray) throws -> MLMultiArray {
        try MLMultiArray(dataPointer: output.dataPointer, shape: output.shape, dataType: output.dataType,
                         strides: output.strides, deallocator: { [self, output] _ in
                             withExtendedLifetime((self, output)) {}
                         })
    }
}
