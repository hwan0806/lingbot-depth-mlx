import json
from pathlib import Path
import shutil

import numpy as np
import pytest

import validate


def test_public_fixtures_are_complete_and_hash_pinned(tmp_path):
    assert len(validate.verify_fixtures()) == 3
    root = tmp_path / 'fixtures'
    shutil.copytree(validate.FIXTURES, root)
    target = root / 'public/corridor_01-sparse-500.npz'
    target.write_bytes(target.read_bytes() + b'tampered')
    with pytest.raises(ValueError, match='checksum'):
        validate.verify_fixtures(root)


@pytest.mark.parametrize('name', ['../escape', '/absolute', 'a/../b', 'a\\b', './a', 'a//b'])
def test_manifest_paths_cannot_escape(name):
    with pytest.raises(ValueError):
        validate.safe_path(name)


def test_model_checks_manifest_inventory_and_contents(tmp_path, monkeypatch):
    package = tmp_path / 'model.mlpackage'
    package.mkdir()
    weight = package / 'weight.bin'
    weight.write_bytes(b'test weights')
    manifest = tmp_path / 'model-manifest.json'
    manifest.write_text(json.dumps({'schemaVersion': 1, 'profile': 'eco',
                                   'files': {'model.mlpackage/weight.bin': validate.sha256(weight)}}))
    monkeypatch.setattr(validate, 'MANIFEST_SHA', validate.sha256(manifest))
    validate.verify_model(package, manifest)
    extra = package / 'extra'
    extra.write_bytes(b'unexpected')
    with pytest.raises(ValueError, match='inventory'):
        validate.verify_model(package, manifest)
    extra.unlink()
    weight.write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        validate.verify_model(package, manifest)
    weight.unlink()
    weight.symlink_to(manifest)
    with pytest.raises(ValueError, match='symlink'):
        validate.verify_model(package, manifest)
    manifest.write_text('{}')
    with pytest.raises(ValueError, match='manifest'):
        validate.verify_model(package, manifest)


def test_output_comparison_and_range_filter_do_not_hide_depth_errors():
    ref = np.ones((1, 2, 2), np.float32) * 2
    data = {'mlx_depth': ref, 'mlx_mask': ref > 0, 'coreml_depth': ref,
            'coreml_probability': np.ones_like(ref), 'gt': ref, 'valid': ref > 0}
    row = validate.compare_sample(data, ref + .5, np.ones_like(ref))
    assert row['agreement_vs_saved_mlx_fp32']['mae'] == .5
    assert row['agreement_vs_saved_mlx_fp32']['pixel_p99_abs_error_m'] == .5
    assert row['gt']['rmse'] == .5
    assert row['mask_iou_vs_saved_mlx'] == 1
    assert row['predicted_mask_coverage'] == 1
    for bad in (np.nan, np.inf, 0, -1):
        pred = ref.copy()
        pred[0, 0, 0] = bad
        with pytest.raises(ValueError):
            validate.compare_sample(data, pred, np.zeros_like(ref))
    pred = np.ones((1, 4, 4))
    pred[0, 0, 0], pred[0, 0, 2], pred[0, 2, 0] = np.nan, .01, 10.01
    excluded = validate.viewer_exclusions(pred)
    assert excluded['sampled_pixels'] == 4 and excluded['excluded_fraction'] == .75
    assert excluded['over_10m_fraction'] == .25


def test_saved_replay_needs_no_model_and_preserves_existing_output(tmp_path):
    output = tmp_path / 'report.json'
    report = validate.evaluate(tmp_path / 'no-model.mlpackage', output, saved=True)
    assert report['all_saved_outputs_numerically_equal']
    assert len(report['samples']) == 3
    assert report['mode'] == 'saved-output-replay'
    with pytest.raises(FileExistsError):
        validate.evaluate(tmp_path / 'missing', output, saved=True)


def test_install_builds_then_copies_verified_model_without_changing_bundle_id(tmp_path):
    commands = validate.install_commands(tmp_path, '00000000-0000000000000000', 'ABCDEFGHIJ',
                                         validate.BUNDLE_ID, tmp_path / 'build')
    assert commands[0][:2] == ['xcrun', 'xcodebuild']
    assert commands[1][2:5] == ['device', 'install', 'app']
    assert commands[2][2:5] == ['device', 'copy', 'to']
    assert commands[3][2:5] == ['device', 'process', 'launch']
    assert 'Documents/' + validate.COREML_MODEL_ID in commands[2]
    assert validate.BUNDLE_ID in commands[2] and validate.BUNDLE_ID in commands[3]
    with pytest.raises(ValueError):
        validate.install_commands(tmp_path, 'device;bad', 'ABCDEFGHIJ', validate.BUNDLE_ID, tmp_path)


def test_app_is_independent_and_uses_pinned_package_verifier():
    base = Path('ios/Apps/LingBotDepthBench')
    project = (base / 'LingBotDepthBench.xcodeproj/project.pbxproj').read_text()
    view = (base / 'Sources/BetaCaptureView.swift').read_text()
    assert '.f32' not in project and '.mlpackage' not in project
    assert 'CameraBenchmark' not in project and 'Benchmark.' not in view
    assert 'ValidatedModelPackage.validate' in view
    assert 'appending(path: ValidatedModelPackage.modelID)' in view
    assert validate.COREML_MODEL_ID in Path('ios/Packages/LingBotDepthRuntime/Sources/LingBotDepthRuntime/ValidatedModelPackage.swift').read_text()
    assert validate.MANIFEST_SHA in Path('ios/Packages/LingBotDepthRuntime/Sources/LingBotDepthRuntime/ValidatedModelPackage.swift').read_text()
    assert validate.sha256(validate.MODEL_MANIFEST) == validate.MANIFEST_SHA
