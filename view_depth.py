"""Local RGB-D point-cloud comparison with viser."""
import argparse
import json
from pathlib import Path
import threading

import cv2
import numpy as np

from validate import FIXTURES, compare_sample, verify_fixtures


def load_cases(results=None, rgb=None, depth=None, depth_unit="mm"):
    if (rgb is None) != (depth is None):
        raise ValueError("--rgb and --depth must be supplied together")
    if rgb is not None:
        if results is None:
            raise ValueError("Custom RGB-D requires --results from run_mlx.py")
        image = cv2.imread(str(rgb), cv2.IMREAD_COLOR)
        raw = cv2.imread(str(depth), cv2.IMREAD_UNCHANGED)
        if image is None or raw is None or raw.ndim != 2 or image.shape[:2] != raw.shape:
            raise ValueError("RGB and single-channel depth must be readable and aligned")
        if depth_unit not in ("mm", "m"):
            raise ValueError("depth unit must be mm or m")
        with np.load(Path(results) / "prediction.npz", allow_pickle=False) as pred:
            return {"custom": make_case(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                raw.astype(np.float32) / (1000 if depth_unit == "mm" else 1),
                pred["depth"], pred["mask_probability"], "MLX · fresh")}
    samples = verify_fixtures()
    report = json.loads((Path(results) / "report.json").read_text()) if results else None
    if report and report.get("mode") != "fresh-mac-mlx":
        raise ValueError("--results must be a run_mlx.py output directory")
    cases = {}
    for sample in samples:
        with np.load(FIXTURES / "public" / sample["file"], allow_pickle=False) as data:
            if results:
                matches = [s for s in report["samples"] if s["sample_id"] == sample["sample_id"]
                           and s.get("fixture_sha256") == sample["sha256"]]
                if len(matches) != 1:
                    raise ValueError("Result/fixture identity mismatch")
                with np.load(Path(results) / sample["file"], allow_pickle=False) as pred:
                    prediction, probability = pred["depth"].copy(), pred["mask_probability"].copy()
                label = "MLX · fresh"
            else:
                prediction, probability = data["mlx_depth"], data["mlx_mask"].astype(np.float32)
                label = "MLX · saved"
            case = make_case(data["rgb"], data["input_depth_m"], prediction, probability, label)
            height, width = data["input_depth_m"].shape
            gt = np.where(data["valid"][0], data["gt"][0], 0)
            case["depths"]["GT · nearest-resized"] = cv2.resize(gt, (width, height), interpolation=cv2.INTER_NEAREST)
            case["depths"]["Core ML · saved"] = data["coreml_depth"][0].copy()
            case["masks"]["Core ML · saved"] = data["coreml_probability"][0] > .5
            metrics = compare_sample(data, prediction, probability)
            case["metrics"] = (f"MLX vs GT: MAE {metrics['gt']['mae']:.4f} m · "
                               f"RMSE {metrics['gt']['rmse']:.4f} m · AbsRel {metrics['gt']['absrel']:.2%}")
            cases[sample["sample_id"]] = case
    return cases


def make_case(rgb, raw, prediction, probability, label):
    if rgb.dtype != np.uint8 or rgb.shape != (*raw.shape, 3) or raw.ndim != 2:
        raise ValueError("Invalid RGB-D shape")
    if prediction.shape != (1, *raw.shape) or probability.shape != prediction.shape:
        raise ValueError("Prediction must match the RGB-D resolution")
    if not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any():
        raise ValueError("Invalid mask probability")
    return {"rgb": rgb.copy(), "depths": {label: prediction[0].copy(), "Input depth": raw.copy()},
            "masks": {label: probability[0] > .5}, "metrics": ""}


def camera_intrinsics(width, height, intrinsics=None, fov=60):
    if intrinsics is None:
        if not np.isfinite(fov) or not 1 < fov < 179:
            raise ValueError("Invalid horizontal FOV")
        focal = width / (2 * np.tan(np.deg2rad(fov) / 2))
        return focal, focal, (width - 1) / 2, (height - 1) / 2
    values = np.asarray(intrinsics, dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all() or (values[:2] <= 0).any():
        raise ValueError("Intrinsics must be finite fx fy cx cy with positive focal lengths")
    return tuple(values)


def point_cloud(depth, rgb, intrinsics, stride=2, max_depth=50, mask=None):
    if stride < 1 or not np.isfinite(max_depth) or max_depth <= .01:
        raise ValueError("Invalid stride/depth range")
    if depth.ndim != 2 or rgb.shape != (*depth.shape, 3):
        raise ValueError("RGB/depth shape mismatch")
    if mask is not None and mask.shape != depth.shape:
        raise ValueError("Mask/depth shape mismatch")
    fx, fy, cx, cy = camera_intrinsics(depth.shape[1], depth.shape[0], intrinsics)
    z = depth[::stride, ::stride]
    v, u = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    finite = np.isfinite(z)
    invalid = ~finite | (z <= .01)
    far = finite & (z > max_depth)
    keep = ~(invalid | far)
    masked = np.zeros_like(keep) if mask is None else keep & ~mask[::stride, ::stride]
    keep &= ~masked
    # OpenCV camera axes -> right-handed viewer: +X right, +Y up, -Z forward.
    points = np.stack(((u[keep] - cx) * z[keep] / fx,
                       -(v[keep] - cy) * z[keep] / fy, -z[keep]), axis=-1).astype(np.float32)
    stats = {"shown": int(keep.sum()), "total": int(keep.size),
             "invalid": float(invalid.mean()), "far": float(far.mean()), "masked": float(masked.mean())}
    return points, rgb[::stride, ::stride][keep], stats


def depth_image(depth, max_depth):
    finite = np.isfinite(depth) & (depth > .01)
    scaled = np.clip(np.where(finite, depth, 0) / max_depth, 0, 1)
    image = cv2.cvtColor(cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    image[~finite] = 0
    return image


def serve(cases, port=8080, intrinsics=None):
    import viser
    if not cases or not 1 <= port <= 65535:
        raise ValueError("Expected cases and a port in 1..65535")
    for case in cases.values():
        h, w = case["rgb"].shape[:2]
        camera_intrinsics(w, h, intrinsics)
    server = viser.ViserServer(host="127.0.0.1", port=port, label="LingBot-Depth")
    server.scene.set_up_direction("+y")
    server.gui.add_markdown("## LingBot-Depth\nRGB-D → RGB point cloud")
    sample = server.gui.add_dropdown("Sample", tuple(cases))
    source = server.gui.add_dropdown("Depth source", tuple(cases[sample.value]["depths"]))
    coloring = server.gui.add_dropdown("Point colors", ("RGB", "Depth"))
    far = server.gui.add_slider("Max depth (m)", min=.1, max=100, step=.1, initial_value=50.)
    stride = server.gui.add_slider("Pixel stride", min=1, max=8, step=1, initial_value=2)
    size = server.gui.add_slider("Point size", min=.001, max=.05, step=.001, initial_value=.01)
    use_mask = server.gui.add_checkbox("Apply model mask", initial_value=False)
    fov = server.gui.add_slider("Approx. horizontal FOV", min=20., max=120., step=1., initial_value=60., disabled=intrinsics is not None)
    server.gui.add_markdown("**Calibration:** " + ("supplied fx/fy/cx/cy in input pixels." if intrinsics else
        "approximate camera (60° by default). No intrinsics in public fixtures; 3D shape is illustrative, not calibrated measurement."))
    reset = server.gui.add_button("Reset view")
    status = server.gui.add_markdown("")
    metrics = server.gui.add_markdown("")
    with server.gui.add_folder("2D preview"):
        rgb_view = server.gui.add_image(cases[sample.value]["rgb"], label="RGB")
        depth_view = server.gui.add_image(cases[sample.value]["rgb"], label="Selected depth")

    def reset_camera(client):
        client.camera.position = (0., 0., 1.)
        client.camera.look_at = (0., 0., -3.)
        client.camera.up_direction = (0., 1., 0.)

    server.on_client_connect(reset_camera)
    reset.on_click(lambda _: [reset_camera(c) for c in server.get_clients().values()])

    def update(_=None):
        case = cases[sample.value]
        if source.value not in case["depths"]:
            source.value = next(iter(case["depths"]))
        d = case["depths"][source.value]
        k = camera_intrinsics(d.shape[1], d.shape[0], intrinsics, fov.value)
        color = case["rgb"] if coloring.value == "RGB" else depth_image(d, far.value)
        mask = case["masks"].get(source.value) if use_mask.value else None
        points, colors, counts = point_cloud(d, color, k, stride.value, far.value, mask)
        with server.atomic():
            server.scene.add_point_cloud("/depth", points, colors, point_size=size.value, point_shape="circle", precision="float32")
            rgb_view.image = case["rgb"]
            depth_view.image = depth_image(d, far.value)
            status.content = (f"**{source.value}** · {counts['shown']:,} / {counts['total']:,} sampled points\n\n"
                f"Excluded: invalid {counts['invalid']:.2%} · beyond range {counts['far']:.2%} · mask {counts['masked']:.2%}")
            metrics.content = case["metrics"]

    def change_sample(event):
        source.options = tuple(cases[sample.value]["depths"])
        update(event)

    sample.on_update(change_sample)
    for control in (source, coloring, far, stride, size, use_mask, fov):
        control.on_update(update)
    update()
    print(f"Open http://127.0.0.1:{server.get_port()} — Ctrl+C to stop.", flush=True)
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, help="run_mlx.py output directory; otherwise show saved public outputs")
    parser.add_argument("--rgb", type=Path)
    parser.add_argument("--depth", type=Path)
    parser.add_argument("--depth-unit", choices=("mm", "m"), default="mm")
    parser.add_argument("--intrinsics", type=float, nargs=4, metavar=("FX", "FY", "CX", "CY"))
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    server = serve(load_cases(args.results, args.rgb, args.depth, args.depth_unit), args.port, args.intrinsics)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
