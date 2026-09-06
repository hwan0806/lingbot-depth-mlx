"""Exercise production 3D projection, and guard the button-only snapshot route."""
import os
from pathlib import Path
import subprocess


def test_beta_cloud_projection_and_capture_contract(tmp_path):
    source = Path('ios/Apps/LingBotDepthBench/Sources/BetaCaptureView.swift').read_text()
    cloud = 'struct BetaCloud:' + source.split('struct BetaCloud:', 1)[1].split('private actor BetaInference', 1)[0]
    swift = 'import Foundation\nimport simd\n' + cloud + r'''
let k = simd_float3x3(rows: [SIMD3<Float>(2, 0, 0.5), SIMD3<Float>(0, 2, 0.5), SIMD3<Float>(0, 0, 1)])
assert(BetaCloud.point(x: 0, y: 0, depth: 2, inverseK: k.inverse) == SIMD3<Float>(0, 0, -2))
assert(BetaCloud.point(x: 1, y: 1, depth: 2, inverseK: k.inverse) == SIMD3<Float>(1, -1, -2))
for d: Float in [.nan, .infinity, -.infinity, -1, 0, 0.01, 10.01] {
    assert(BetaCloud.point(x: 1, y: 1, depth: d, inverseK: k.inverse) == nil)
}
assert(BetaCloud.point(x: 0, y: 0, depth: 10, inverseK: k.inverse) != nil)
assert(BetaCloud.color(SIMD3<Float>(1, 0, 0)) == SIMD4<Float>(1, 0, 0, 1))
assert(BetaCloud.color(SIMD3<Float>(0.2, 0.4, 0.8)) == SIMD4<Float>(0.2, 0.4, 0.8, 1))
assert(BetaCloud.color(SIMD3<Float>(-1, 2, 0.5)) == SIMD4<Float>(0, 1, 0.5, 1))
assert(BetaCloud.color(SIMD3<Float>(.nan, 0, 0)) == nil)
assert(BetaCloud.color(SIMD3<Float>(0, .infinity, 0)) == nil)
'''
    subprocess.run(['xcrun', 'swift', '-'], input=swift, text=True, check=True,
                   env=dict(os.environ, DEVELOPER_DIR='/Applications/Xcode.app/Contents/Developer'))
    capture = source.split('    func capture()', 1)[1].split('    private static func makeScene', 1)[0]
    assert capture.index('ARFrameAdapter.makeFrame') < capture.index('task = Task')
    assert 'guard ready, !busy, active, hasDepth' in capture
    assert 'worker.infer(frame)' in capture and 'Task.checkCancellation()' in capture
    draw = source.split('        func draw(in view:', 1)[1]
    assert 'predict(' not in draw and '.infer(' not in draw
    assert source.count('backend.predict(') == 1
    assert 'view.preferredFramesPerSecond = 30' in source
    assert 'task?.cancel()' in source and 'session.pause()' in source
    assert 'lidarColors.append(color)' in source and 'colors.append(color)' in source
    assert 'input.depthMeters[[0, 0, NSNumber(value: y), NSNumber(value: x)]]' in source
    toggle = source.split('// Toggle only geometry visibility;', 1)[1].split('\n    }', 1)[0]
    assert '"lidar"' in toggle and '"lingbot"' in toggle
    assert 'pointOfView =' not in toggle and 'view.scene =' not in toggle


def test_depth_preview_filter_has_finite_sensor_extent():
    source = Path('ios/Apps/LingBotDepthBench/Sources/BetaCaptureView.swift').read_text()
    pipeline = 'let depthImage = ' + source.split('let depthImage = ', 1)[1].split('            let bounds = ', 1)[0]
    pipeline = pipeline.replace('CIImage(cvPixelBuffer: depth.depthMap)', 'CIImage(color: .white).cropped(to: depthBounds)')
    swift = '''import CoreImage
import ImageIO
let depthBounds = CGRect(x: 0, y: 0, width: 256, height: 192)
let orientation = CGImagePropertyOrientation.up
''' + pipeline + '''
assert(!depthImage.extent.isInfinite && depthImage.extent == depthBounds)
assert(CIContext().createCGImage(depthImage, from: depthImage.extent) != nil)
'''
    subprocess.run(['xcrun', 'swift', '-'], input=swift, text=True, check=True,
                   env=dict(os.environ, DEVELOPER_DIR='/Applications/Xcode.app/Contents/Developer'))
