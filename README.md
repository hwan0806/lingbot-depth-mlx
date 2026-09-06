<div align="center">

# LingBot-Depth · Core ML Validation

**iPad capture app + public RGB-D fixtures + reproducible depth checks**

No model implementation. No weight conversion. No MLX or PyTorch dependency.

</div>

## Included / excluded

| Included | Excluded |
|---|---|
| iPad RGB/LiDAR preview and RGB 3D comparison app | MLX model port and Transformer/decoder implementation |
| Swift + Metal runtime for an existing Core ML model | Core ML exporters, quantizers, optimization experiments |
| 3 public iBims RGB-D/GT cases and saved outputs | Model weights, original checkpoints, private captures |
| GT/agreement metrics, SHA-256 checks, tests and installation CLI | Original porting repository or its Git history |

**Model upload / download: TBD.** Supply the validated **safe107 Core ML `.mlpackage`** separately. A PyTorch `.pt`, MLX safetensors, or `weight.bin` alone is not sufficient.

## Quick start · no weights needed

Apple Silicon Mac, Python 3.10–3.13, and `uv`. The included `uv.lock` pins the environment; no other repository is required.

```bash
uv sync --extra test
uv run python validate.py verify
uv run python validate.py run --saved --output outputs/saved.json
```

This verifies fixture hashes and recomputes metrics from **saved Core ML outputs**. It does not execute a model or regenerate the saved PyTorch/MLX references.

## Test supplied weights

Place the complete package at `models/model.mlpackage`, or pass another path:

```bash
uv run python validate.py verify --model models/model.mlpackage
uv run python validate.py run --model models/model.mlpackage \
  --output outputs/fresh.json
```

The package is checked against the [pinned model manifest](models/model-manifest.json) before loading. Missing, extra, modified, or symlinked package files are rejected. Model files and generated outputs are ignored by Git; existing reports are not overwritten.

| Check | Report |
|---|---|
| Agreement vs. saved MLX FP32 | MAE, RMSE, AbsRel, pixel p99/max error, mask IoU |
| GT accuracy | Native-resolution valid GT; prediction resized with bilinear interpolation |
| Coverage | Finite positive depth, predicted mask, viewer-excluded depth fractions |
| Reproduction | Maximum depth/mask difference vs. archived Core ML outputs |

Fresh inference uses **CPU+GPU**, Eco1200, 560 × 420. The report identifies saved replay versus new inference explicitly. A successful command means validation completed—not that a new model passed a quality-acceptance gate. `all_saved_outputs_numerically_equal` reports exact numerical reproduction separately.

The manifest SHA-256 is:

```text
ee3ceba8609ff768e55217979b6226c9034a627133c769f6ede7d058361fdccf
```

This pins the original package files, **not** a compiled `.mlmodelc`. Other models are intentionally rejected; they need their own reviewed manifest and references, not a disabled hash check.

## iPad app

**30 Hz RGB/LiDAR preview → capture once → RGB point cloud → LingBot / LiDAR toggle**

- One aligned RGB/depth/camera snapshot; one Core ML inference per button press.
- Shared RGB and camera coordinates; orbit, zoom, and viewpoint-preserving source switching.
- Viewer samples every 2 pixels and displays depths in **(0.01, 10] m**. This filter can hide out-of-range predictions; the validation report lists the excluded fractions.
- RGB-D refinement, not RGB-only inference. No scanning fusion, mesh export, or continuous 30 Hz model inference.

Requirements: full Xcode, a LiDAR-equipped iPad on iPadOS 18+, Developer Mode, an unlocked/trusted device, and your Xcode signing account.

```bash
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
xcrun xctrace list devices

# Set these locally to your hardware UDID and signing Team ID:
export LINGBOT_IPAD_UDID="YOUR_HARDWARE_UDID"
export LINGBOT_SIGNING_TEAM="YOUR_TEAM_ID"

uv run python validate.py install --model models/model.mlpackage \
  --device "$LINGBOT_IPAD_UDID" --team "$LINGBOT_SIGNING_TEAM" --dry-run

uv run python validate.py install --model models/model.mlpackage \
  --device "$LINGBOT_IPAD_UDID" --team "$LINGBOT_SIGNING_TEAM"
```

**Build → install app → copy verified model to Documents → launch.** The app verifies the package hashes again, then compiles/loads Core ML. Grant camera access and wait for readiness.

The [Xcode project](ios/Apps/LingBotDepthBench/LingBotDepthBench.xcodeproj) builds without bundled weights or private fixture resources. No Team ID is embedded in this standalone project. The original Bundle ID is retained; use `--bundle-id com.yourname.LingBotDepthBench` if your signing team needs another ID. A changed ID creates a separate app/data container. Installing with the same ID updates the existing app and model payload without uninstalling it.

## Public data & evidence

The ~8 MB [fixture set](fixtures/public) contains three **deliberately selected** iBims cases:

| Sample | Selection reason |
|---|---|
| `lab_11-sparse-500` | Highest GT AbsRel in the historical 60-case run |
| `corridor_01-sparse-500` | Largest GT pixel error |
| `corridor_01-lowres-8x` | Largest MLX/Core ML agreement error |

[60-case metrics](evidence/full60-metrics.json) · [Historical summary](evidence/safe107-evidence.json) · [Earlier release reproduction](evidence/safe107-reproduction.json) · [Standalone reproduction](evidence/standalone-reproduction.json)

- **Not 60 independent scenes:** the full evaluation used 20 images × 3 synthetic corruptions.
- **Not real LiDAR GT:** input depth was synthesized from laser-scanner GT. The historical iPad summary uses a different private frame; that frame is not included.
- **GT and agreement differ:** the 18.48 m GT outlier also occurs in original PyTorch output. At that pixel, GT is 34.618 m, PyTorch 16.149 m, Core ML 16.143 m. Nearby GT is about 15–17 m; a GT anomaly is possible but unconfirmed.
- **Viewer filtering:** reported excluded fractions apply the viewer's depth filter to these public cases, not a live iPad observation.

Historical device results do not certify this newly separated app build. Physical-device installation/retest is pending; model upload remains TBD.

## Tests

```bash
uv run pytest
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcrun swift test --package-path ios/Packages/LingBotDepthRuntime
```

Python checks cover metrics, fixture/model integrity, viewer filtering, installation command construction, and production app projection/preview logic. Swift checks cover camera geometry and model-package validation. No conversion or training is performed.

Verified in this standalone repository: **29 Python tests, 2 Swift tests, and an unsigned iOS Release build passed**. Fresh Mac Core ML inference on all three public fixtures exactly reproduced the saved depth/mask outputs in an environment without MLX or PyTorch. This does not replace an iPad retest.

## Credits & licenses

Code: [Apache-2.0](LICENSE). Runtime and comparison code were extracted from the Apple-platform implementation; model-port and conversion sources are not included.

Model: [LingBot-Depth](https://huggingface.co/robbyant/lingbot-depth-pretrain-vitl-14-v0.5), by the LingBot-Depth authors. Existing exported weights only; no official affiliation is claimed.

Data: [iBims-v1](https://mediatum.ub.tum.de/1455541), Tobias Koch, Lukas Liebel, Friedrich Fraundorfer, Marco Körner (2019), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). See [attribution and modifications](fixtures/public/ATTRIBUTION.md).
