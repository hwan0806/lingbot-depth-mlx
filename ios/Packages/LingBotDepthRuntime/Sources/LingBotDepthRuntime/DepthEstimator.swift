public protocol DepthEstimating: Sendable {
    var profile: DepthProfile { get }
    var kind: DepthBackendKind { get }
    func load() async throws
    func prewarm() async throws
    func predict(_ input: ModelTensors) async throws -> DepthResult
    func unload() async
}
