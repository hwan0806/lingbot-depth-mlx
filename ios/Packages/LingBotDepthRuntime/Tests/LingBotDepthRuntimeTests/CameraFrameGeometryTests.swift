import CoreVideo
import ImageIO
import simd
import Testing
@testable import LingBotDepthRuntime

@Test func cameraFrameConvertsARKitOriginAndAxesAtTheBoundary() throws {
    var pixels: CVPixelBuffer?
    try #require(CVPixelBufferCreate(nil, 4, 4, kCVPixelFormatType_32BGRA, nil, &pixels) == kCVReturnSuccess)
    let buffer = try #require(pixels)
    let k = simd_float3x3(rows: [SIMD3(100, 2, 1), SIMD3(0, 120, 2), SIMD3(0, 0, 1)])
    func frame(_ pose: simd_float4x4) -> CameraFrame {
        CameraFrame(rgb: buffer, depth: buffer, confidence: nil, intrinsics: k,
                    cameraTransform: pose, orientation: .up, timestamp: 0)
    }
    let previous = frame(matrix_identity_float4x4)
    #expect(previous.intrinsics[2, 0] == 1.5)
    #expect(previous.intrinsics[2, 1] == 2.5)
    // A center-origin landmark and its edge-origin counterpart must yield the same ray.
    let expectedRay = k.inverse * SIMD3<Float>(0, 0, 1)
    let actualRay = previous.intrinsics.inverse * SIMD3<Float>(0.5, 0.5, 1)
    #expect(simd_length(expectedRay - actualRay) < 1e-6)
    let orientations: [(CGImagePropertyOrientation, SIMD3<Float>)] = [
        (.up, SIMD3(1, 1, 1)), (.upMirrored, SIMD3(7, 1, 1)),
        (.down, SIMD3(7, 7, 1)), (.downMirrored, SIMD3(1, 7, 1)),
        (.leftMirrored, SIMD3(1, 1, 1)), (.right, SIMD3(7, 1, 1)),
        (.rightMirrored, SIMD3(7, 7, 1)), (.left, SIMD3(1, 7, 1)),
    ]
    for (orientation, firstCenter) in orientations {
        let geometry = FrameGeometry(aspectFill: CGSize(width: 4, height: 4),
                                     modelSize: CGSize(width: 8, height: 8),
                                     intrinsics: previous.intrinsics, orientation: orientation)
        #expect(simd_length(geometry.intrinsics.inverse * firstCenter - expectedRay) < 1e-6)
    }
    #expect(previous.cameraTransform * SIMD4<Float>(1, 2, 3, 1) == SIMD4<Float>(1, -2, -3, 1))

    // Camera moves right 0.25m, up 1m, forward 1m in ARKit world axes.
    var moved = matrix_identity_float4x4
    moved.columns.3 = SIMD4(0.25, 1, -1, 1)
    let current = frame(moved)
    let relative = current.cameraTransform.inverse * previous.cameraTransform
    let point = relative * SIMD4<Float>(0, 0, 2, 1)
    #expect(simd_length(point - SIMD4<Float>(-0.25, 1, 1, 1)) < 1e-6)

    // Independent 90-degree yaw oracle, including a point now behind the camera.
    let yaw = simd_float4x4(columns: (SIMD4(0, 0, -1, 0), SIMD4(0, 1, 0, 0),
                                    SIMD4(1, 0, 0, 0), SIMD4(0, 0, 0, 1)))
    let rotated = frame(yaw)
    let rotatedPoint = rotated.cameraTransform.inverse * previous.cameraTransform * SIMD4<Float>(1, 1, 2, 1)
    #expect(simd_length(rotatedPoint - SIMD4<Float>(2, 1, -1, 1)) < 1e-6)
}
