import CoreML
import Foundation
import Metal

public enum DepthProfile: String, Codable, Sendable {
    case eco, balanced, quality
    case ecoLite768 = "eco-lite-768"
    case ecoMid972 = "eco-mid-972"
    case ecoLite588 = "eco-lite-588"

    public var shape: (height: Int, width: Int) {
        switch self {
        case .eco: (420, 560)
        case .ecoLite768: (336, 448)
        case .ecoMid972: (378, 504)
        case .ecoLite588: (294, 392)
        case .balanced: (518, 686)
        case .quality: (728, 966)
        }
    }
}

public enum DepthBackendKind: String, Codable, Sendable { case coreML }

public struct ModelTensors: @unchecked Sendable {
    public let image: MLMultiArray
    public let depthMeters: MLMultiArray
    public let validity: MLMultiArray
    public let timestamp: TimeInterval

    public init(image: MLMultiArray, depthMeters: MLMultiArray, validity: MLMultiArray, timestamp: TimeInterval) {
        self.image = image
        self.depthMeters = depthMeters
        self.validity = validity
        self.timestamp = timestamp
    }
}

public struct DepthResult: @unchecked Sendable {
    public let depth: MLMultiArray
    public let maskProbability: MLMultiArray?
    public let profile: DepthProfile
    public let backend: DepthBackendKind
    public let latency: Duration
    public let timestamp: TimeInterval
    public let depthBuffer: MTLBuffer?
    public let maskBuffer: MTLBuffer?

    public init(depth: MLMultiArray, maskProbability: MLMultiArray?, profile: DepthProfile, backend: DepthBackendKind, latency: Duration, timestamp: TimeInterval, depthBuffer: MTLBuffer? = nil, maskBuffer: MTLBuffer? = nil) {
        self.depth = depth
        self.maskProbability = maskProbability
        self.profile = profile
        self.backend = backend
        self.latency = latency
        self.timestamp = timestamp
        self.depthBuffer = depthBuffer
        self.maskBuffer = maskBuffer
    }
}

public enum DepthRuntimeError: Error {
    case modelNotLoaded
    case missingOutput(String)
    case unsupportedProfile(DepthProfile)
}

func makeSharedMultiArray(device: MTLDevice, shape: [Int], dataType: MLMultiArrayDataType = .float32) throws -> (MLMultiArray, MTLBuffer) {
    let count = shape.reduce(1, *)
    let byteCount = switch dataType {
    case .float16: count * MemoryLayout<Float16>.size
    case .float32: count * MemoryLayout<Float>.size
    default: throw MetalError.unsupportedPixelFormat
    }
    guard let buffer = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
        throw MetalError.bufferCreation
    }
    var stride = 1
    var strides = Array(repeating: 1, count: shape.count)
    for index in shape.indices.reversed() {
        strides[index] = stride
        stride *= shape[index]
    }
    let array = try MLMultiArray(
        dataPointer: buffer.contents(),
        shape: shape.map(NSNumber.init),
        dataType: dataType,
        strides: strides.map(NSNumber.init),
        deallocator: { _ in withExtendedLifetime(buffer) {} }
    )
    return (array, buffer)
}
