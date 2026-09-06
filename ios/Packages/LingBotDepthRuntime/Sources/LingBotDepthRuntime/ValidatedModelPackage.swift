import CryptoKit
import Foundation

/// Validate a supplied Core ML package; no conversion or fallback model.
public enum ValidatedModelPackage {
    public static let manifestSHA256 = "ee3ceba8609ff768e55217979b6226c9034a627133c769f6ede7d058361fdccf"

    private struct Manifest: Decodable {
        let schemaVersion: Int
        let profile: String
        let files: [String: String]
    }

    public enum ValidationError: LocalizedError {
        case invalidPackage
        public var errorDescription: String? {
            "Missing or invalid safe107 model. Install the verified model payload into Documents/safe107-model."
        }
    }

    public static func sha256(_ file: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: file)
        defer { try? handle.close() }
        var digest = SHA256()
        while let chunk = try handle.read(upToCount: 1_048_576), !chunk.isEmpty {
            digest.update(data: chunk)
        }
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
    }

    public static func validate(at directory: URL, expectedManifestSHA256: String = manifestSHA256) throws -> URL {
        guard try directory.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink != true else {
            throw ValidationError.invalidPackage
        }
        let root = directory.resolvingSymlinksInPath()
        let manifestURL = root.appending(path: "model-manifest.json")
        guard manifestURL.resolvingSymlinksInPath().path == manifestURL.path,
              try sha256(manifestURL) == expectedManifestSHA256 else { throw ValidationError.invalidPackage }
        let manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: manifestURL))
        guard manifest.schemaVersion == 1, manifest.profile == "eco", !manifest.files.isEmpty else {
            throw ValidationError.invalidPackage
        }
        let package = root.appending(path: "model.mlpackage")
        guard package.resolvingSymlinksInPath().path == package.path,
              let files = FileManager.default.enumerator(at: package,
                includingPropertiesForKeys: [.isRegularFileKey, .isSymbolicLinkKey]) else {
            throw ValidationError.invalidPackage
        }
        var actual = Set<String>()
        for case let file as URL in files {
            let properties = try file.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
            guard properties.isSymbolicLink != true else { throw ValidationError.invalidPackage }
            if properties.isRegularFile == true {
                // FileManager may enumerate /private/var while Foundation normalizes to /var.
                let components = file.resolvingSymlinksInPath().pathComponents
                guard components.starts(with: root.pathComponents) else { throw ValidationError.invalidPackage }
                actual.insert(components.dropFirst(root.pathComponents.count).joined(separator: "/"))
            }
        }
        guard actual == Set(manifest.files.keys) else { throw ValidationError.invalidPackage }
        for (name, expected) in manifest.files {
            let components = name.split(separator: "/", omittingEmptySubsequences: false)
            guard name.hasPrefix("model.mlpackage/"), !name.contains("\\"),
                  !components.contains(where: { $0.isEmpty || $0 == "." || $0 == ".." }) else {
                throw ValidationError.invalidPackage
            }
            let file = root.appending(path: name)
            guard file.resolvingSymlinksInPath().path == file.path, try sha256(file) == expected else {
                throw ValidationError.invalidPackage
            }
        }
        return package.resolvingSymlinksInPath()
    }
}
