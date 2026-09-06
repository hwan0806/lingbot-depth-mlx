import Foundation
import Testing
@testable import LingBotDepthRuntime

@Test func modelPackageValidatesEveryFileAndRefusesTampering() throws {
    let root = FileManager.default.temporaryDirectory.appending(path: "coreml-package-check-\(UUID().uuidString)")
    let package = root.appending(path: "model.mlpackage")
    try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let weights = package.appending(path: "weight.bin")
    try Data("test weights".utf8).write(to: weights)
    let manifestURL = root.appending(path: "model-manifest.json")
    let manifest: [String: Any] = ["schemaVersion": 1, "profile": "eco",
                                  "files": ["model.mlpackage/weight.bin": try ValidatedModelPackage.sha256(weights)]]
    try JSONSerialization.data(withJSONObject: manifest).write(to: manifestURL)
    let digest = try ValidatedModelPackage.sha256(manifestURL)
    #expect(try ValidatedModelPackage.validate(at: root, expectedManifestSHA256: digest) == package.resolvingSymlinksInPath())
    #expect(throws: (any Error).self) { try ValidatedModelPackage.validate(at: root) }
    let unexpected = package.appending(path: "unexpected")
    try Data().write(to: unexpected)
    #expect(throws: (any Error).self) { try ValidatedModelPackage.validate(at: root, expectedManifestSHA256: digest) }
    try FileManager.default.removeItem(at: unexpected)
    try Data("tampered".utf8).write(to: weights)
    #expect(throws: (any Error).self) { try ValidatedModelPackage.validate(at: root, expectedManifestSHA256: digest) }
    try FileManager.default.removeItem(at: weights)
    try FileManager.default.createSymbolicLink(at: weights, withDestinationURL: manifestURL)
    #expect(throws: (any Error).self) { try ValidatedModelPackage.validate(at: root, expectedManifestSHA256: digest) }
}
