#if os(iOS) && canImport(ARKit)
import ARKit
import ImageIO

public enum ARFrameAdapter {
    public static func makeFrame(from frame: ARFrame, orientation: CGImagePropertyOrientation, smoothed: Bool = true, presentationEpoch: UInt64 = 0) -> CameraFrame? {
        guard let sceneDepth = smoothed ? (frame.smoothedSceneDepth ?? frame.sceneDepth) : frame.sceneDepth else { return nil }
        return CameraFrame(
            rgb: frame.capturedImage,
            depth: sceneDepth.depthMap,
            confidence: sceneDepth.confidenceMap,
            intrinsics: frame.camera.intrinsics,
            cameraTransform: frame.camera.transform,
            orientation: orientation,
            timestamp: frame.timestamp,
            presentationEpoch: presentationEpoch
        )
    }
}
#endif
