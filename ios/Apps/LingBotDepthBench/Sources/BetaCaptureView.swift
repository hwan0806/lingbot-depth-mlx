import ARKit
import AVFoundation
import CoreImage
import CoreML
import LingBotDepthRuntime
import MetalKit
import SceneKit
import SwiftUI

struct BetaCaptureView: View {
    @StateObject private var model = BetaCaptureModel()
    @State private var showLidar = false
    @Environment(\.scenePhase) private var scenePhase

    var body: some View {
        VStack(spacing: 12) {
            HStack {
                Text("LingBot · Capture Beta").font(.title2.bold())
                Spacer()
                Text(String(format: "미리보기 %.1f Hz · 목표 30", model.previewHz)).font(.caption.monospacedDigit())
            }
            HStack { Text("카메라 RGB"); Spacer(); Text("LiDAR 원본 깊이 · 0–5m") }
                .font(.headline)
            BetaPreview(model: model).frame(maxHeight: .infinity)
                .clipShape(RoundedRectangle(cornerRadius: 12))
                .accessibilityLabel("실시간 카메라와 LiDAR 깊이 미리보기")
            HStack {
                Button(action: model.capture) {
                    Label(model.busy ? "LingBot 추론 중…" : "현재 순간 추출 → 3D", systemImage: "camera.aperture")
                        .font(.headline).padding(8)
                }
                .buttonStyle(.borderedProminent)
                .disabled(!model.ready || !model.hasDepth || model.busy || scenePhase != .active)
                .accessibilityIdentifier("captureDepth")
                if model.busy { ProgressView() }
                Spacer()
                if model.scene != nil {
                    Button("3D 초기 시점") { model.resetView += 1 }
                }
            }
            Text(model.status).font(.callout).frame(maxWidth: .infinity, alignment: .leading)
                .accessibilityIdentifier("captureStatus")
            if let scene = model.scene {
                Picker("3D 비교", selection: $showLidar) {
                    Text("LingBot + RGB").tag(false)
                    Text("LiDAR + RGB").tag(true)
                }.pickerStyle(.segmented).accessibilityIdentifier("cloudSource")
                BetaCloudView(scene: scene, showLidar: showLidar).id(model.resetView).frame(maxHeight: .infinity)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                    .accessibilityLabel("\(showLidar ? "LiDAR" : "LingBot") RGB 포인트 클라우드. 드래그로 회전, 두 손가락으로 확대")
                Text("같은 촬영·RGB·좌표계 · 전환 시 시점 유지 · LiDAR는 모델 입력 격자에 정렬 · 단위 m")
                    .font(.caption).foregroundStyle(.secondary)
            } else {
                ContentUnavailableView("버튼을 눌러 3D 깊이 추출", systemImage: "cube.transparent",
                    description: Text("연속 추론하지 않습니다. 촬영 순간의 RGB와 LiDAR를 한 쌍으로 처리합니다."))
                    .frame(maxHeight: .infinity)
            }
            if !model.ready && !model.loading {
                Button("카메라 / 모델 다시 준비") { Task { await model.start() } }
            }
        }
        .padding().background(Color(uiColor: .systemGroupedBackground))
        .task { await model.start() }
        .onChange(of: scenePhase) { _, phase in
            if phase == .active { model.resume() } else { model.pause() }
        }
        .onDisappear { model.pause() }
    }
}

@MainActor final class BetaCaptureModel: ObservableObject {
    let session = ARSession()
    private let worker = BetaInference()
    @Published var ready = false
    @Published var loading = false
    @Published var hasDepth = false
    @Published var busy = false
    @Published var status = "카메라와 Core ML Eco1200 · FP16/FP32 모델 준비 중…"
    @Published var scene: SCNScene?
    @Published var resetView = 0
    @Published var previewHz = 0.0
    private var active = true
    private var lastFrameTimestamp: Double?
    private var lastFrameArrival = -Double.infinity
    var previewActive: Bool { active }
    private var task: Task<Void, Never>?

    func observeFrame(_ timestamp: Double) {
        if lastFrameTimestamp != timestamp {
            lastFrameTimestamp = timestamp
            lastFrameArrival = CACurrentMediaTime()
        }
        let fresh = active && CACurrentMediaTime() - lastFrameArrival < 1
        if hasDepth != fresh { hasDepth = fresh }
    }

    func start() async {
        guard !loading, !ready else { return }
        loading = true
        defer { loading = false }
        do {
            guard await AVCaptureDevice.requestAccess(for: .video) else {
                throw BetaError.message("설정에서 카메라 접근을 허용한 뒤 다시 시도하세요.")
            }
            guard ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) else {
                throw BetaError.message("LiDAR sceneDepth 지원 iPad가 필요합니다.")
            }
            resume()
            try await worker.load()
            ready = true
            status = "준비 완료 · Core ML Eco1200 · FP16/FP32 · CPU+GPU · 버튼을 누르면 1회 추론"
        } catch { status = error.localizedDescription }
    }

    func resume() {
        active = true
        guard ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth),
              AVCaptureDevice.authorizationStatus(for: .video) == .authorized else { return }
        let configuration = ARWorldTrackingConfiguration()
        if let format = ARWorldTrackingConfiguration.supportedVideoFormats.first(where: {
            $0.framesPerSecond == 30 && $0.captureDevicePosition == .back
        }) { configuration.videoFormat = format }
        configuration.frameSemantics = .sceneDepth
        session.run(configuration)
        UIApplication.shared.isIdleTimerDisabled = true
    }

    func pause() {
        active = false
        session.pause()
        hasDepth = false
        task?.cancel()
        UIApplication.shared.isIdleTimerDisabled = false
    }

    func capture() {
        guard ready, !busy, active, hasDepth, CACurrentMediaTime() - lastFrameArrival < 1,
              let ar = session.currentFrame,
              let frame = ARFrameAdapter.makeFrame(from: ar, orientation: .up, smoothed: false) else {
            status = "최신 LiDAR 프레임을 기다려 주세요."
            return
        }
        // Retain exactly this RGB/depth/K/pose tuple before creating any asynchronous work.
        busy = true
        status = "순간 고정 완료 · LingBot 깊이 추론 중…"
        task = Task {
            defer { busy = false }
            do {
                let cloud = try await worker.infer(frame)
                try Task.checkCancellation()
                guard active else { return }
                scene = Self.makeScene(cloud)
                resetView += 1
                status = String(format: "완료 · 추론 %.2f초 / 전체 %.2f초 · LingBot %d / LiDAR %d points", cloud.inferenceSeconds, cloud.totalSeconds, cloud.positions.count, cloud.lidarPositions.count)
            } catch is CancellationError {
                status = "촬영 처리를 취소했습니다. 다시 촬영할 수 있습니다."
            } catch { status = "추출 실패: \(error.localizedDescription)" }
        }
    }

    private static func makeScene(_ cloud: BetaCloud) -> SCNScene {
        let scene = SCNScene()
        scene.rootNode.addChildNode(makeCloudNode(positions: cloud.positions, colors: cloud.colors, name: "lingbot"))
        scene.rootNode.addChildNode(makeCloudNode(positions: cloud.lidarPositions, colors: cloud.lidarColors, name: "lidar"))
        let camera = SCNNode(); camera.camera = SCNCamera()
        camera.camera?.zNear = 0.01; camera.camera?.zFar = 100
        camera.position = SCNVector3(0, 0, 1)
        camera.look(at: SCNVector3(0, 0, -2))
        scene.rootNode.addChildNode(camera)
        return scene
    }

    private static func makeCloudNode(positions: [SIMD3<Float>], colors: [SIMD4<Float>], name: String) -> SCNNode {
        let vertices = positions.withUnsafeBytes { Data($0) }
        let colorData = colors.withUnsafeBytes { Data($0) }
        let sources = [
            SCNGeometrySource(data: vertices, semantic: .vertex, vectorCount: positions.count,
                usesFloatComponents: true, componentsPerVector: 3, bytesPerComponent: 4, dataOffset: 0, dataStride: MemoryLayout<SIMD3<Float>>.stride),
            SCNGeometrySource(data: colorData, semantic: .color, vectorCount: colors.count,
                usesFloatComponents: true, componentsPerVector: 4, bytesPerComponent: 4, dataOffset: 0, dataStride: MemoryLayout<SIMD4<Float>>.stride)
        ]
        let element = SCNGeometryElement(indices: positions.indices.map(UInt32.init), primitiveType: .point)
        element.pointSize = 3; element.minimumPointScreenSpaceRadius = 1; element.maximumPointScreenSpaceRadius = 4
        let geometry = SCNGeometry(sources: sources, elements: [element])
        let material = SCNMaterial(); material.lightingModel = .constant; material.isDoubleSided = true
        geometry.materials = [material]
        let node = SCNNode(geometry: geometry)
        node.name = name; node.isHidden = name == "lidar"
        return node
    }
}

private enum BetaError: LocalizedError {
    case message(String)
    var errorDescription: String? { switch self { case .message(let message): message } }
}

struct BetaCloud: Sendable {
    var positions: [SIMD3<Float>]
    var colors: [SIMD4<Float>]
    var lidarPositions: [SIMD3<Float>]
    var lidarColors: [SIMD4<Float>]
    var inferenceSeconds: Double
    var totalSeconds: Double

    static func color(_ rgb: SIMD3<Float>) -> SIMD4<Float>? {
        guard rgb.x.isFinite, rgb.y.isFinite, rgb.z.isFinite else { return nil }
        let c = simd_clamp(rgb, .zero, SIMD3(repeating: 1))
        return SIMD4(c.x, c.y, c.z, 1)
    }

    static func point(x: Int, y: Int, depth: Float, inverseK: simd_float3x3) -> SIMD3<Float>? {
        guard depth.isFinite, depth > 0.01, depth <= 10 else { return nil }
        let ray = inverseK * SIMD3(Float(x) + 0.5, Float(y) + 0.5, 1)
        let p = ray * depth
        guard p.x.isFinite, p.y.isFinite, p.z.isFinite else { return nil }
        return SIMD3(p.x, -p.y, -p.z) // Optical +Y down/+Z forward -> SceneKit camera.
    }
}

private actor BetaInference {
    private var backend: CoreMLBackend?
    private var preprocessor: MetalPreprocessor?

    func load() async throws {
        guard backend == nil else { return }
        guard let device = MTLCreateSystemDefaultDevice() else { throw BetaError.message("Metal을 사용할 수 없습니다.") }
        let package = try ValidatedModelPackage.validate(at: URL.documentsDirectory.appending(path: ValidatedModelPackage.modelID))
        let model = CoreMLBackend(modelURL: package, profile: .eco, placement: .cpuAndGPU)
        try await model.load()
        preprocessor = try MetalPreprocessor(device: device)
        backend = model
    }

    func infer(_ frame: CameraFrame) async throws -> BetaCloud {
        guard let backend, let preprocessor else { throw BetaError.message("모델이 아직 준비되지 않았습니다.") }
        let start = ContinuousClock.now
        let input = try await preprocessor.encode(frame, profile: .eco)
        try Task.checkCancellation()
        let output = try await backend.predict(input)
        try Task.checkCancellation()
        let size = DepthProfile.eco.shape
        let geometry = FrameGeometry(aspectFill: CGSize(width: CVPixelBufferGetWidth(frame.rgb), height: CVPixelBufferGetHeight(frame.rgb)),
            modelSize: CGSize(width: size.width, height: size.height), intrinsics: frame.intrinsics, orientation: frame.orientation)
        let inverseK = geometry.intrinsics.inverse
        var positions: [SIMD3<Float>] = [], colors: [SIMD4<Float>] = []
        var lidarPositions: [SIMD3<Float>] = [], lidarColors: [SIMD4<Float>] = []
        // ponytail: render every second pixel (~59k points); use GPU geometry if beta interaction is slow.
        for y in stride(from: 0, to: size.height, by: 2) {
            try Task.checkCancellation()
            for x in stride(from: 0, to: size.width, by: 2) {
                // The existing preprocessor supplies aligned RGB in [0,1], not normalized network activations.
                let rgb = SIMD3<Float>(
                    input.image[[0, 0, NSNumber(value: y), NSNumber(value: x)]].floatValue,
                    input.image[[0, 1, NSNumber(value: y), NSNumber(value: x)]].floatValue,
                    input.image[[0, 2, NSNumber(value: y), NSNumber(value: x)]].floatValue)
                guard let color = BetaCloud.color(rgb) else { throw BetaError.message("촬영 RGB에 NaN/Inf가 있습니다.") }
                let lidar = input.depthMeters[[0, 0, NSNumber(value: y), NSNumber(value: x)]].floatValue
                if let p = BetaCloud.point(x: x, y: y, depth: lidar, inverseK: inverseK) {
                    lidarPositions.append(p); lidarColors.append(color)
                }
                let value = output.depth[[0, NSNumber(value: y), NSNumber(value: x)]].floatValue
                guard value.isFinite else { throw BetaError.message("모델 깊이에 NaN/Inf가 있습니다.") }
                if let p = BetaCloud.point(x: x, y: y, depth: value, inverseK: inverseK) {
                    positions.append(p)
                    colors.append(color)
                }
            }
        }
        guard !positions.isEmpty else { throw BetaError.message("표시 가능한 깊이가 없습니다. 다른 장면에서 다시 촬영하세요.") }
        func seconds(_ d: Duration) -> Double { Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18 }
        return BetaCloud(positions: positions, colors: colors, lidarPositions: lidarPositions, lidarColors: lidarColors,
            inferenceSeconds: seconds(output.latency), totalSeconds: seconds(start.duration(to: .now)))
    }
}

private struct BetaCloudView: UIViewRepresentable {
    let scene: SCNScene
    let showLidar: Bool
    func makeUIView(context: Context) -> SCNView {
        let view = SCNView()
        view.backgroundColor = .black
        view.scene = scene; view.allowsCameraControl = true
        view.preferredFramesPerSecond = 30
        view.defaultCameraController.interactionMode = .orbitTurntable
        view.defaultCameraController.target = SCNVector3(0, 0, -2)
        view.pointOfView = scene.rootNode.childNodes.first(where: { $0.camera != nil })
        updateUIView(view, context: context)
        return view
    }
    func updateUIView(_ view: SCNView, context: Context) {
        // Toggle only geometry visibility; keep the interactive camera and view instance unchanged.
        view.scene?.rootNode.childNode(withName: "lidar", recursively: false)?.isHidden = !showLidar
        view.scene?.rootNode.childNode(withName: "lingbot", recursively: false)?.isHidden = showLidar
    }
}

private final class BetaDisplayCount: @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0
    func presented() { lock.withLock { count += 1 } }
    func take() -> Int { lock.withLock { let n = count; count = 0; return n } }
}

private struct BetaPreview: UIViewRepresentable {
    let model: BetaCaptureModel
    func makeCoordinator() -> Renderer { Renderer(model: model) }
    func makeUIView(context: Context) -> MTKView {
        let view = MTKView(frame: .zero, device: MTLCreateSystemDefaultDevice())
        view.framebufferOnly = false; view.colorPixelFormat = .bgra8Unorm
        view.preferredFramesPerSecond = 30
        view.delegate = context.coordinator
        return view
    }
    func updateUIView(_ view: MTKView, context: Context) {}
    static func dismantleUIView(_ view: MTKView, coordinator: Renderer) { view.isPaused = true; view.delegate = nil }

    @MainActor final class Renderer: NSObject, @preconcurrency MTKViewDelegate {
        let model: BetaCaptureModel
        let device = MTLCreateSystemDefaultDevice()
        private lazy var queue = device?.makeCommandQueue()
        private lazy var context = device.map { CIContext(mtlDevice: $0) }
        private let counts = BetaDisplayCount()
        private var lastSample = CACurrentMediaTime()
        init(model: BetaCaptureModel) { self.model = model }
        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}
        func draw(in view: MTKView) {
            guard model.previewActive, let frame = model.session.currentFrame, let depth = frame.sceneDepth,
                  let drawable = view.currentDrawable, let command = queue?.makeCommandBuffer(), let context else { return }
            model.observeFrame(frame.timestamp)
            let orientation: CGImagePropertyOrientation = switch view.window?.windowScene?.interfaceOrientation {
            case .portrait: .right
            case .portraitUpsideDown: .left
            case .landscapeLeft: .down
            default: .up
            }
            let rgb = CIImage(cvPixelBuffer: frame.capturedImage).oriented(orientation)
            let depthBounds = CGRect(x: 0, y: 0, width: CVPixelBufferGetWidth(depth.depthMap), height: CVPixelBufferGetHeight(depth.depthMap))
            let depthImage = CIImage(cvPixelBuffer: depth.depthMap)
                .applyingFilter("CIColorMatrix", parameters: [
                    "inputRVector": CIVector(x: 0.2, y: 0, z: 0, w: 0),
                    "inputGVector": CIVector(x: 0.2, y: 0, z: 0, w: 0),
                    "inputBVector": CIVector(x: 0.2, y: 0, z: 0, w: 0),
                    "inputAVector": CIVector(x: 0, y: 0, z: 0, w: 0),
                    "inputBiasVector": CIVector(x: 0, y: 0, z: 0, w: 1)])
                .applyingFilter("CIFalseColor", parameters: ["inputColor0": CIColor(red: 0, green: 0.7, blue: 1), "inputColor1": CIColor(red: 1, green: 0.1, blue: 0)])
                // Alpha bias expands Core Image's extent to infinity; restore the sensor raster before fitting.
                .cropped(to: depthBounds)
                .oriented(orientation)
            let bounds = CGRect(origin: .zero, size: view.drawableSize)
            func fit(_ image: CIImage, x: CGFloat) -> CIImage {
                let cell = CGSize(width: bounds.width / 2, height: bounds.height)
                let s = min(cell.width / image.extent.width, cell.height / image.extent.height)
                return image.transformed(by: CGAffineTransform(translationX: -image.extent.minX, y: -image.extent.minY))
                    .transformed(by: CGAffineTransform(scaleX: s, y: s))
                    .transformed(by: CGAffineTransform(translationX: x + (cell.width - image.extent.width * s) / 2, y: (cell.height - image.extent.height * s) / 2))
            }
            let composite = fit(rgb, x: 0).composited(over: fit(depthImage, x: bounds.width / 2))
                .composited(over: CIImage(color: .black).cropped(to: bounds))
            context.render(composite, to: drawable.texture, commandBuffer: command, bounds: bounds, colorSpace: CGColorSpaceCreateDeviceRGB())
            drawable.addPresentedHandler { [counts] d in if d.presentedTime > 0 { counts.presented() } }
            command.present(drawable); command.commit()
            let now = CACurrentMediaTime()
            if now - lastSample >= 1 {
                model.previewHz = Double(counts.take()) / (now - lastSample)
                lastSample = now
            }
        }
    }
}
