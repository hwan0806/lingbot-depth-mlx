#!/usr/bin/env python3
"""Validate supplied Core ML weights or replay public reference outputs. No model conversion."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
COREML_MODEL_ID = 'lingbot-depth-coreml-eco1200-mixed-fp16-fp32'
MANIFEST_SHA = 'ee3ceba8609ff768e55217979b6226c9034a627133c769f6ede7d058361fdccf'
MODEL_MANIFEST = ROOT / 'models/model-manifest.json'
FIXTURES = ROOT / 'fixtures'
BUNDLE_ID = 'com.siliconpedia.LingBotDepthBench'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def safe_path(name: str) -> Path:
    path = PurePosixPath(name)
    if not name or path.is_absolute() or '\\' in name or '..' in path.parts or str(path) != name:
        raise ValueError('unsafe manifest path')
    return Path(name)


def verify_files(root: Path, expected: dict) -> None:
    if not root.is_dir() or root.is_symlink():
        raise ValueError('missing or symlinked directory: ' + str(root))
    paths = list(root.rglob('*'))
    if any(p.is_symlink() for p in paths):
        raise ValueError('symlink in verified directory')
    if {str(p.relative_to(root)) for p in paths if p.is_file()} != set(expected):
        raise ValueError('file inventory mismatch')
    for name, digest in expected.items():
        if sha256(root / safe_path(name)) != digest:
            raise ValueError('checksum mismatch: ' + name)


def verify_fixtures(root: Path = FIXTURES) -> list[dict]:
    inventory = json.loads((root / 'manifest.json').read_text())
    verify_files(root / 'public', inventory['files'])
    samples = json.loads((root / 'public/samples.json').read_text())
    if len(samples) != 3 or len({s['sample_id'] for s in samples}) != 3:
        raise ValueError('expected three distinct public cases')
    for sample in samples:
        safe_path(sample['file'])
        if inventory['files'].get(sample['file']) != sample['sha256']:
            raise ValueError('fixture identity mismatch')
    return samples


def verify_model(package: Path, manifest_path: Path = MODEL_MANIFEST) -> None:
    if manifest_path.is_symlink() or sha256(manifest_path) != MANIFEST_SHA:
        raise ValueError('pinned Core ML manifest mismatch')
    manifest = json.loads(manifest_path.read_text())
    if manifest['schemaVersion'] != 1 or manifest['profile'] != 'eco':
        raise ValueError('model manifest contract mismatch')
    files = {}
    for name, digest in manifest['files'].items():
        if not name.startswith('model.mlpackage/'):
            raise ValueError('invalid model member')
        relative = name.removeprefix('model.mlpackage/')
        safe_path(relative)
        files[relative] = digest
    verify_files(package, files)


def viewer_exclusions(depth) -> dict:
    import numpy as np
    values = np.asarray(depth)[0, ::2, ::2]
    finite = np.isfinite(values)
    invalid = ~finite | (values <= .01)
    far = finite & (values > 10)
    return {'sampled_pixels': values.size, 'nonfinite_fraction': float((~finite).mean()),
            'at_or_below_001m_fraction': float((finite & (values <= .01)).mean()),
            'over_10m_fraction': float(far.mean()), 'excluded_fraction': float((invalid | far).mean()),
            'scope': 'Depth filter only, stride 2, no model-mask filtering; not a live iPad observation.'}


def compare_sample(data, depth, probability) -> dict:
    import cv2
    import numpy as np
    from metrics import depth_metrics, mask_iou

    ref = data['mlx_depth'].astype(np.float64)
    if depth.shape != ref.shape or probability.shape != ref.shape or not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any():
        raise ValueError('invalid output shape/mask probability')
    agreement = depth_metrics(ref, depth)
    agreement['pixel_p99_abs_error_m'] = float(np.percentile(np.abs(depth-ref), 99, method='inverted_cdf'))
    gt, valid = data['gt'].astype(np.float64), data['valid']
    native = cv2.resize(depth[0], (gt.shape[-1], gt.shape[-2]), interpolation=cv2.INTER_LINEAR)[None]
    gt_metrics = depth_metrics(gt, native, valid)
    gt_metrics['pixel_p99_abs_error_m'] = float(np.percentile(np.abs(native-gt)[valid], 99, method='inverted_cdf'))
    return {'agreement_vs_saved_mlx_fp32': agreement, 'gt': gt_metrics,
            'mask_iou_vs_saved_mlx': mask_iou(data['mlx_mask'], probability > .5),
            'predicted_mask_coverage': float((probability > .5).mean()),
            'finite_positive_depth_coverage': float((np.isfinite(depth) & (depth > 0)).mean()),
            'max_depth_difference_vs_saved_coreml_m': float(np.max(np.abs(depth.astype(np.float64)-data['coreml_depth']))),
            'max_mask_difference_vs_saved_coreml': float(np.max(np.abs(probability.astype(np.float64)-data['coreml_probability']))),
            'viewer_exclusions': viewer_exclusions(depth)}


def evaluate(package: Path, output: Path, saved: bool = False) -> dict:
    import numpy as np
    if output.exists():
        raise FileExistsError(output)
    samples = verify_fixtures()
    model = None
    if not saved:
        verify_model(package)
        import coremltools as ct
        model = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_AND_GPU)
        shapes = {x.name: tuple(x.type.multiArrayType.shape) for x in model.get_spec().description.input}
        if shapes != {'image': (1, 3, 420, 560), 'depth_m': (1, 1, 420, 560), 'validity': (1, 1, 420, 560)}:
            raise ValueError('model input contract mismatch')
        if set(model.output_description) != {'select_1', 'sigmoid'}:
            raise ValueError('model output contract mismatch')
    rows = []
    for sample in samples:
        with np.load(FIXTURES / 'public' / sample['file'], allow_pickle=False) as data:
            if model is None:
                depth, probability = data['coreml_depth'], data['coreml_probability']
            else:
                d = data['input_depth_m'][None, None].astype(np.float32)
                pred = model.predict({'image': data['rgb'].transpose(2, 0, 1)[None].astype(np.float32)/255,
                                      'depth_m': d, 'validity': (d > .01).astype(np.float32)})
                depth, probability = pred['select_1'], pred['sigmoid']
            rows.append({'sample_id': sample['sample_id'], 'fixture_sha256': sample['sha256'],
                         **compare_sample(data, depth, probability)})
    result = {'mode': 'saved-output-replay' if saved else 'fresh-mac-coreml-cpu-and-gpu',
              'model_id': COREML_MODEL_ID, 'model_manifest_sha256': MANIFEST_SHA, 'samples': rows,
              'environment': {'python': platform.python_version(), 'macos': platform.mac_ver()[0],
                              'architecture': platform.machine(),
                              'packages': {p: version(p) for p in ('numpy', 'opencv-python', 'coremltools')}},
              'all_saved_outputs_numerically_equal': all(r['max_depth_difference_vs_saved_coreml_m'] == 0 and
                                                        r['max_mask_difference_vs_saved_coreml'] == 0 for r in rows),
              'scope': 'Three selected public cases, not all 60 cases, iPad accuracy, or a speed benchmark. No MLX/PyTorch model execution.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return result


def install_commands(payload: Path, device: str, team: str, bundle_id: str, build: Path) -> list[list[str]]:
    if not re.fullmatch('[A-Za-z0-9-]+', device) or not re.fullmatch('[A-Z0-9]{10}', team):
        raise ValueError('invalid device/team identifier')
    if not re.fullmatch(r'[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+', bundle_id):
        raise ValueError('invalid bundle identifier')
    app = build / 'Build/Products/Release-iphoneos/LingBotDepthBench.app'
    return [
        ['xcrun', 'xcodebuild', '-project', str(ROOT / 'ios/Apps/LingBotDepthBench/LingBotDepthBench.xcodeproj'),
         '-scheme', 'LingBotDepthBench', '-configuration', 'Release', '-destination', 'id=' + device,
         '-derivedDataPath', str(build), '-allowProvisioningUpdates', 'DEVELOPMENT_TEAM=' + team,
         'PRODUCT_BUNDLE_IDENTIFIER=' + bundle_id, 'build'],
        ['xcrun', 'devicectl', 'device', 'install', 'app', '--device', device, str(app)],
        ['xcrun', 'devicectl', 'device', 'copy', 'to', '--device', device,
         '--source', str(payload), '--destination', 'Documents/' + COREML_MODEL_ID,
         '--domain-type', 'appDataContainer', '--domain-identifier', bundle_id],
        ['xcrun', 'devicectl', 'device', 'process', 'launch', '--device', device,
         '--terminate-existing', bundle_id],
    ]


def install(package: Path, device: str, team: str, bundle_id: str, dry_run: bool) -> None:
    verify_model(package)
    build = ROOT / 'build/ipad'
    if dry_run:
        for command in install_commands(Path('<temporary-model-payload>') / COREML_MODEL_ID, device, team, bundle_id, build):
            print(json.dumps(command))
        return
    build.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='model-upload-', dir=build) as stage:
        payload = Path(stage) / COREML_MODEL_ID
        payload.mkdir()
        shutil.copytree(package, payload / 'model.mlpackage')
        shutil.copyfile(MODEL_MANIFEST, payload / 'model-manifest.json')
        verify_model(payload / 'model.mlpackage', payload / 'model-manifest.json')
        for command in install_commands(payload, device, team, bundle_id, build):
            subprocess.run(command, check=True, env=os.environ.copy())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('verify', 'run', 'install'):
        p = sub.add_parser(name)
        p.add_argument('--model', type=Path, default=None if name == 'verify' else ROOT / 'models' / COREML_MODEL_ID / 'model.mlpackage')
        if name == 'run':
            p.add_argument('--saved', action='store_true', help='Replay saved Core ML outputs; no weights required')
            p.add_argument('--output', type=Path, required=True)
        if name == 'install':
            p.add_argument('--device', required=True, help='Hardware UDID, not the CoreDevice UUID')
            p.add_argument('--team', required=True)
            p.add_argument('--bundle-id', default=BUNDLE_ID)
            p.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.command == 'verify':
        verify_fixtures()
        if args.model is not None:
            verify_model(args.model)
    elif args.command == 'run':
        report = evaluate(args.model, args.output, saved=args.saved)
        print(json.dumps({'mode': report['mode'], 'saved_outputs_equal': report['all_saved_outputs_numerically_equal']}))
    else:
        install(args.model, args.device, args.team, args.bundle_id, args.dry_run)
    print(json.dumps({'command': args.command, 'status': 'completed'}))


if __name__ == '__main__':
    main()
