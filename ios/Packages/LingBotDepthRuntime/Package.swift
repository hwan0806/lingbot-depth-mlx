// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "LingBotDepthRuntime",
    platforms: [.iOS(.v18), .macOS(.v15)],
    products: [.library(name: "LingBotDepthRuntime", targets: ["LingBotDepthRuntime"])],
    targets: [
        .target(
            name: "LingBotDepthRuntime",
            resources: [.process("Shaders")]
        ),
        .testTarget(name: "LingBotDepthRuntimeTests", dependencies: ["LingBotDepthRuntime"]),
    ]
)
