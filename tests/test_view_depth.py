import json

import cv2
import numpy as np
import pytest

from view_depth import camera_intrinsics, depth_image, load_cases, point_cloud, serve


def test_projection_preserves_rgb_and_reports_filtered_points():
    depth = np.array([[2, 4, np.nan], [0, 60, 3]], dtype=np.float32)
    rgb = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    points, colors, stats = point_cloud(depth, rgb, (2, 2, 0, 0), stride=1, max_depth=50)
    np.testing.assert_allclose(points, [[0, 0, -2], [2, 0, -4], [3, -1.5, -3]])
    np.testing.assert_array_equal(colors, rgb[[0, 0, 1], [0, 1, 2]])
    assert stats["shown"] == 3 and stats["total"] == 6
    assert stats["invalid"] == pytest.approx(2 / 6)
    assert stats["far"] == pytest.approx(1 / 6)
    mask = np.ones_like(depth, dtype=bool)
    mask[0, 0] = False
    assert point_cloud(depth, rgb, (2, 2, 0, 0), 1, 50, mask)[2]["masked"] == pytest.approx(1 / 6)
    empty, colors, _ = point_cloud(np.zeros((2, 3)), rgb, (2, 2, 0, 0))
    assert empty.shape == colors.shape == (0, 3)


def test_intrinsics_and_depth_color():
    assert camera_intrinsics(100, 80, (50, 60, 49, 39)) == (50, 60, 49, 39)
    fx, fy, cx, cy = camera_intrinsics(100, 80, fov=90)
    np.testing.assert_allclose([fx, fy, cx, cy], [50, 50, 49.5, 39.5])
    with pytest.raises(ValueError):
        camera_intrinsics(100, 80, (0, 50, 0, 0))
    with pytest.raises(ValueError):
        camera_intrinsics(100, 80, fov=np.nan)
    rgb = depth_image(np.array([[0, np.nan, 2.]]), 50)
    assert rgb.dtype == np.uint8 and not rgb[0, :2].any() and rgb[0, 2].any()


def test_public_saved_cases_and_result_identity(tmp_path):
    cases = load_cases()
    assert len(cases) == 3
    for case in cases.values():
        assert set(case["depths"]) == {"Input depth", "MLX · saved", "Core ML · saved", "GT · nearest-resized"}
        assert all(d.shape == case["rgb"].shape[:2] for d in case["depths"].values())
        assert case["metrics"].startswith("MLX vs GT:")
    (tmp_path / "report.json").write_text(json.dumps({"mode": "fresh-mac-mlx", "samples": []}))
    with pytest.raises(ValueError, match="identity"):
        load_cases(tmp_path)


def test_custom_rgbd_and_invalid_prediction(tmp_path):
    rgb = np.full((4, 6, 3), [10, 20, 30], dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(tmp_path / "depth.png"), np.full((4, 6), 2000, dtype=np.uint16))
    np.savez(tmp_path / "prediction.npz", depth=np.full((1, 4, 6), 3.), mask_probability=np.ones((1, 4, 6)))
    cases = load_cases(tmp_path, tmp_path / "rgb.png", tmp_path / "depth.png")
    np.testing.assert_array_equal(cases["custom"]["rgb"], rgb)
    np.testing.assert_allclose(cases["custom"]["depths"]["Input depth"], 2)
    np.savez(tmp_path / "prediction.npz", depth=np.ones((1, 2, 2)), mask_probability=np.ones((1, 2, 2)))
    with pytest.raises(ValueError, match="resolution"):
        load_cases(tmp_path, tmp_path / "rgb.png", tmp_path / "depth.png")
    with pytest.raises(ValueError, match="together"):
        load_cases(rgb="only.png")


def test_viser_http_smoke():
    pytest.importorskip("viser")
    import socket
    from urllib.request import urlopen
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = serve(load_cases(), port=port)
    try:
        assert server.get_host() == "127.0.0.1"
        with urlopen(f"http://127.0.0.1:{server.get_port()}", timeout=5) as response:
            assert response.status == 200 and b"<html" in response.read().lower()
    finally:
        server.stop()
