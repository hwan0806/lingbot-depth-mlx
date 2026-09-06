"""Run preconverted MLX weights on public fixtures or an aligned RGB-D pair."""
import argparse
import json
from pathlib import Path
import platform
import time
from importlib.metadata import version

import numpy as np

from validate import FIXTURES, compare_sample, sha256, verify_fixtures


def run(model_path: Path, output: Path, rgb=None, depth=None, depth_unit="mm") -> dict:
    if output.exists():
        raise FileExistsError(output)
    if (rgb is None) != (depth is None):
        raise ValueError("--rgb and --depth must be supplied together")
    samples = verify_fixtures() if rgb is None else None
    import mlx.core as mx
    from lingbot_depth_mlx.model import LingBotDepth
    from lingbot_depth_mlx.preprocessing import load_rgb, load_depth

    started = time.perf_counter()
    model = LingBotDepth.from_pretrained(model_path)
    model.eval()
    mx.eval(model.parameters())
    load_seconds = time.perf_counter() - started
    output.mkdir(parents=True, exist_ok=False)
    rows = []

    def predict(rgb_array, depth_array, name):
        started = time.perf_counter()
        prediction = model.infer(rgb_array, depth_array, num_tokens=1200)
        mx.eval(prediction)
        seconds = time.perf_counter() - started
        arrays = {k: np.asarray(v) for k, v in prediction.items()}
        if not all(np.isfinite(v).all() for v in arrays.values()):
            raise ValueError("MLX produced nonfinite output")
        np.savez_compressed(output / (name + ".npz"), **arrays)
        return arrays, seconds

    if samples is None:
        arrays, seconds = predict(load_rgb(rgb), load_depth(depth, depth_unit), "prediction")
        rows.append({"sample_id": "custom", "inference_seconds": seconds,
                     "depth_shape": list(arrays["depth"].shape)})
    else:
        for sample in samples:
            with np.load(FIXTURES / "public" / sample["file"], allow_pickle=False) as data:
                arrays, seconds = predict(data["rgb"], data["input_depth_m"], sample["sample_id"])
                rows.append({"sample_id": sample["sample_id"], "fixture_sha256": sample["sha256"],
                             "inference_seconds": seconds,
                             **compare_sample(data, arrays["depth"], arrays["mask_probability"])})
    report = {
        "mode": "fresh-mac-mlx", "num_tokens": 1200,
        "precision": "bf16-encoder-fp32-decoder" if (model_path / "precision.json").exists() else "fp32",
        "weights_sha256": sha256(model_path / "weights.safetensors"),
        "model_load_seconds": load_seconds, "samples": rows,
        "environment": {"python": platform.python_version(), "macos": platform.mac_ver()[0],
                        "architecture": platform.machine(), "mlx": version("mlx")},
        "scope": "Single passes including preprocessing and mx.eval, not a controlled benchmark. "
                 "Agreement uses saved MLX FP32; GT is synthetic public RGB-D, not iPad LiDAR.",
    }
    with (output / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rgb", type=Path)
    parser.add_argument("--depth", type=Path)
    parser.add_argument("--depth-unit", choices=("mm", "m"), default="mm")
    parser.add_argument("--view", action="store_true", help="Open a local viser server after inference")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--intrinsics", type=float, nargs=4, metavar=("FX", "FY", "CX", "CY"))
    args = parser.parse_args()
    if args.view:
        from importlib.util import find_spec
        if find_spec("viser") is None:
            parser.error("--view requires: uv sync --extra mlx --extra viewer")
    result = run(args.model, args.output, args.rgb, args.depth, args.depth_unit)
    print(json.dumps({"status": "completed", "samples": len(result["samples"]),
                      "report": str(args.output / "report.json")}))
    if args.view:
        import threading
        from view_depth import load_cases, serve
        server = serve(load_cases(args.output, args.rgb, args.depth, args.depth_unit), args.port, args.intrinsics)
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
        finally:
            server.stop()


if __name__ == "__main__":
    main()
