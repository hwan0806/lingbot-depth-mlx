import CoreGraphics
import CoreVideo
import Foundation
import ImageIO
import simd

public struct CameraFrame: @unchecked Sendable {
    public let rgb: CVPixelBuffer
    public let depth: CVPixelBuffer
    public let confidence: CVPixelBuffer?
    /// Source-image K in edge-origin coordinates (pixel centers are n + 0.5).
    public let intrinsics: simd_float3x3
    /// Optical camera (+X right, +Y down, +Z forward) to ARKit world.
    public let cameraTransform: simd_float4x4
    public let orientation: CGImagePropertyOrientation
    public let timestamp: TimeInterval
    public let presentationEpoch: UInt64

    /// Accepts native ARKit K (upper-left pixel center origin) and camera-to-world pose.
    /// Normalize once here; callers must not pass already-normalized matrices.
    public init(rgb: CVPixelBuffer, depth: CVPixelBuffer, confidence: CVPixelBuffer?, intrinsics: simd_float3x3, cameraTransform: simd_float4x4, orientation: CGImagePropertyOrientation, timestamp: TimeInterval, presentationEpoch: UInt64 = 0) {
        self.rgb = rgb
        self.depth = depth
        self.confidence = confidence
        let centerToEdge = simd_float3x3(rows: [SIMD3(1, 0, 0.5), SIMD3(0, 1, 0.5), SIMD3(0, 0, 1)])
        self.intrinsics = centerToEdge * intrinsics
        self.cameraTransform = cameraTransform * simd_float4x4(diagonal: SIMD4(1, -1, -1, 1))
        self.orientation = orientation
        self.timestamp = timestamp
        self.presentationEpoch = presentationEpoch
    }
}

/// Serializes geometry invalidation against the final synchronous GPU submission.
public final class PresentationEpochGate: @unchecked Sendable {
    private let lock = NSLock()
    private var epoch: UInt64 = 0
    public init() {}
    public var current: UInt64 { lock.withLock { epoch } }
    @discardableResult public func invalidate() -> UInt64 {
        lock.withLock { epoch += 1; return epoch }
    }
    public func isCurrent(_ value: UInt64) -> Bool { lock.withLock { value == epoch } }
    /// The action must be short and must not reenter this gate or await GPU completion.
    @discardableResult public func performIfCurrent(_ value: UInt64, _ action: () throws -> Void) rethrows -> Bool {
        try lock.withLock {
            guard value == epoch else { return false }
            try action()
            return true
        }
    }
}
