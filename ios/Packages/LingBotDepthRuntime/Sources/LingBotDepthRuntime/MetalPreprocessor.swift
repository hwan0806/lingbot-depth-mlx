import CoreML
import CoreVideo
import Foundation
import Metal
import os

private let preprocessLog = OSLog(subsystem: "com.siliconpedia.LingBotDepth", category: .pointsOfInterest)

public final class MetalPreprocessor: @unchecked Sendable {
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState
    private var textureCache: CVMetalTextureCache?

    public init(device: MTLDevice) throws {
        self.device = device
        guard let queue = device.makeCommandQueue() else { throw MetalError.unavailable }
        self.queue = queue
        let library = try device.makeDefaultLibrary(bundle: .module)
        guard let function = library.makeFunction(name: "preprocessYCbCrDepth") else { throw MetalError.missingFunction }
        self.pipeline = try device.makeComputePipelineState(function: function)
        CVMetalTextureCacheCreate(nil, nil, device, nil, &textureCache)
    }

    public func encode(_ frame: CameraFrame, profile: DepthProfile) async throws -> ModelTensors {
        os_signpost(.begin, log: preprocessLog, name: "Preprocess")
        defer { os_signpost(.end, log: preprocessLog, name: "Preprocess") }
        let shape = profile.shape
        let imageShape = [1, 3, shape.height, shape.width]
        let depthShape = [1, 1, shape.height, shape.width]
        let (image, imageBuffer) = try makeSharedMultiArray(device: device, shape: imageShape)
        let (depth, depthBuffer) = try makeSharedMultiArray(device: device, shape: depthShape)
        let (validity, validityBuffer) = try makeSharedMultiArray(device: device, shape: depthShape)
        guard CVPixelBufferGetPlaneCount(frame.rgb) == 2,
              CVPixelBufferGetPixelFormatType(frame.depth) == kCVPixelFormatType_DepthFloat32,
              let y = texture(frame.rgb, pixelFormat: .r8Unorm, plane: 0),
              let cbcr = texture(frame.rgb, pixelFormat: .rg8Unorm, plane: 1),
              let sourceDepth = texture(frame.depth, pixelFormat: .r32Float, plane: 0),
              let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else {
            throw MetalError.unsupportedPixelFormat
        }
        let geometry = FrameGeometry(
            aspectFill: CGSize(width: CVPixelBufferGetWidth(frame.rgb), height: CVPixelBufferGetHeight(frame.rgb)),
            modelSize: CGSize(width: shape.width, height: shape.height),
            intrinsics: frame.intrinsics,
            orientation: frame.orientation
        )
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(y.texture, index: 0)
        encoder.setTexture(cbcr.texture, index: 1)
        encoder.setTexture(sourceDepth.texture, index: 2)
        encoder.setBuffer(imageBuffer, offset: 0, index: 0)
        encoder.setBuffer(depthBuffer, offset: 0, index: 1)
        encoder.setBuffer(validityBuffer, offset: 0, index: 2)
        var outputSize = SIMD2<UInt32>(UInt32(shape.width), UInt32(shape.height))
        encoder.setBytes(&outputSize, length: MemoryLayout.size(ofValue: outputSize), index: 3)
        var modelToSource = geometry.modelToSource
        var sourceSize = SIMD2<Float>(Float(geometry.sourceSize.width), Float(geometry.sourceSize.height))
        encoder.setBytes(&modelToSource, length: MemoryLayout.size(ofValue: modelToSource), index: 4)
        encoder.setBytes(&sourceSize, length: MemoryLayout.size(ofValue: sourceSize), index: 5)
        let threads = MTLSize(width: 16, height: 16, depth: 1)
        encoder.dispatchThreads(MTLSize(width: shape.width, height: shape.height, depth: 1), threadsPerThreadgroup: threads)
        encoder.endEncoding()
        await command.commitAndAwaitCompletion()
        if let error = command.error { throw error }
        return ModelTensors(image: image, depthMeters: depth, validity: validity, timestamp: frame.timestamp)
    }

    private func texture(_ buffer: CVPixelBuffer, pixelFormat: MTLPixelFormat, plane: Int) -> (wrapper: CVMetalTexture, texture: MTLTexture)? {
        guard let textureCache else { return nil }
        let width = CVPixelBufferGetWidthOfPlane(buffer, plane)
        let height = CVPixelBufferGetHeightOfPlane(buffer, plane)
        var wrapped: CVMetalTexture?
        CVMetalTextureCacheCreateTextureFromImage(nil, textureCache, buffer, nil, pixelFormat, width, height, plane, &wrapped)
        guard let wrapped, let texture = CVMetalTextureGetTexture(wrapped) else { return nil }
        return (wrapped, texture)
    }
}

public enum MetalError: Error { case unavailable, missingFunction, bufferCreation, unsupportedPixelFormat, missingSharedOutput }

extension MTLCommandBuffer {
    func commitAndAwaitCompletion() async {
        await withCheckedContinuation { continuation in
            addCompletedHandler { _ in continuation.resume() }
            commit()
        }
    }
}
