import CoreGraphics
import ImageIO
import simd

public struct FrameGeometry: Sendable {
    public let sourceSize: CGSize
    public let orientedSize: CGSize
    public let crop: CGRect
    public let modelSize: CGSize
    public let intrinsics: simd_float3x3
    public let modelToSource: simd_float3x3

    public init(sourceSize: CGSize, crop: CGRect, modelSize: CGSize, intrinsics: simd_float3x3, orientation: CGImagePropertyOrientation = .up) {
        self.sourceSize = sourceSize
        let orientationTransform = Self.orientationTransform(orientation, sourceSize: sourceSize)
        self.orientedSize = orientation.swapsAxes ? CGSize(width: sourceSize.height, height: sourceSize.width) : sourceSize
        self.crop = crop
        self.modelSize = modelSize
        let sx = Float(modelSize.width / crop.width)
        let sy = Float(modelSize.height / crop.height)
        let cropTransform = simd_float3x3(rows: [
            SIMD3(sx, 0, -Float(crop.minX) * sx),
            SIMD3(0, sy, -Float(crop.minY) * sy),
            SIMD3(0, 0, 1),
        ])
        let sourceToModel = cropTransform * orientationTransform
        self.intrinsics = sourceToModel * intrinsics
        self.modelToSource = sourceToModel.inverse
    }

    public init(aspectFill sourceSize: CGSize, modelSize: CGSize, intrinsics: simd_float3x3, orientation: CGImagePropertyOrientation) {
        let oriented = orientation.swapsAxes ? CGSize(width: sourceSize.height, height: sourceSize.width) : sourceSize
        let sourceAspect = oriented.width / oriented.height
        let targetAspect = modelSize.width / modelSize.height
        let crop: CGRect
        if sourceAspect > targetAspect {
            let width = oriented.height * targetAspect
            crop = CGRect(x: (oriented.width - width) / 2, y: 0, width: width, height: oriented.height)
        } else {
            let height = oriented.width / targetAspect
            crop = CGRect(x: 0, y: (oriented.height - height) / 2, width: oriented.width, height: height)
        }
        self.init(sourceSize: sourceSize, crop: crop, modelSize: modelSize, intrinsics: intrinsics, orientation: orientation)
    }

    private static func orientationTransform(_ orientation: CGImagePropertyOrientation, sourceSize: CGSize) -> simd_float3x3 {
        let w = Float(sourceSize.width)
        let h = Float(sourceSize.height)
        let rows: [SIMD3<Float>] = switch orientation {
        case .up: [SIMD3(1, 0, 0), SIMD3(0, 1, 0), SIMD3(0, 0, 1)]
        case .upMirrored: [SIMD3(-1, 0, w), SIMD3(0, 1, 0), SIMD3(0, 0, 1)]
        case .down: [SIMD3(-1, 0, w), SIMD3(0, -1, h), SIMD3(0, 0, 1)]
        case .downMirrored: [SIMD3(1, 0, 0), SIMD3(0, -1, h), SIMD3(0, 0, 1)]
        case .leftMirrored: [SIMD3(0, 1, 0), SIMD3(1, 0, 0), SIMD3(0, 0, 1)]
        case .right: [SIMD3(0, -1, h), SIMD3(1, 0, 0), SIMD3(0, 0, 1)]
        case .rightMirrored: [SIMD3(0, -1, h), SIMD3(-1, 0, w), SIMD3(0, 0, 1)]
        case .left: [SIMD3(0, 1, 0), SIMD3(-1, 0, w), SIMD3(0, 0, 1)]
        @unknown default: [SIMD3(1, 0, 0), SIMD3(0, 1, 0), SIMD3(0, 0, 1)]
        }
        return simd_float3x3(rows: rows)
    }
}

private extension CGImagePropertyOrientation {
    var swapsAxes: Bool {
        switch self {
        case .left, .leftMirrored, .right, .rightMirrored: true
        default: false
        }
    }
}
