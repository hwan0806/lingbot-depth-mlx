from pathlib import Path

import numpy as np
import pytest

from lingbot_depth_mlx import weights
from run_mlx import run
from validate import sha256


def test_checkpoint_hashes_and_precision(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "weights.safetensors").write_bytes(b"test weights")
    monkeypatch.setattr(weights, "CONFIG_SHA", sha256(tmp_path / "config.json"))
    digest = sha256(tmp_path / "weights.safetensors")
    monkeypatch.setattr(weights, "WEIGHT_SHA", {"fp32": digest, "bf16": digest})
    assert weights.verify_checkpoint(tmp_path) == "fp32"
    (tmp_path / "precision.json").write_text("{}")
    monkeypatch.setattr(weights, "PRECISION_SHA", sha256(tmp_path / "precision.json"))
    assert weights.verify_checkpoint(tmp_path) == "bf16"
    (tmp_path / "precision.json").write_text("tampered")
    with pytest.raises(ValueError, match="checksum"):
        weights.verify_checkpoint(tmp_path)
    (tmp_path / "precision.json").unlink()
    (tmp_path / "weights.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        weights.verify_checkpoint(tmp_path)
    (tmp_path / "weights.safetensors").unlink()
    (tmp_path / "weights.safetensors").symlink_to(tmp_path / "config.json")
    with pytest.raises(ValueError, match="checksum"):
        weights.verify_checkpoint(tmp_path)
    (tmp_path / "quantization.json").write_text("{}")
    with pytest.raises(ValueError, match="quantized"):
        weights.verify_checkpoint(tmp_path)


def test_run_rejects_overwrite_and_unpaired_input(tmp_path):
    with pytest.raises(FileExistsError):
        run(Path("missing"), tmp_path)
    with pytest.raises(ValueError, match="together"):
        run(Path("missing"), tmp_path / "new", rgb="rgb.png")


def test_mlx_input_geometry_and_invalid_depth():
    pytest.importorskip("mlx.core")
    from lingbot_depth_mlx.preprocessing import preprocess_encoder_inputs, token_grid
    assert token_grid(480, 640, 1200) == (30, 40)
    rgb = np.zeros((14, 28, 3), np.uint8)
    depth = np.full((14, 28), 2, np.float32)
    depth[0, :6] = [0, -1, np.nan, np.inf, -np.inf, .01]
    _, remapped, valid, grid = preprocess_encoder_inputs(rgb, depth, 2)
    assert grid == (1, 2)
    assert not np.asarray(valid)[0, 0, :6, 0].any()
    np.testing.assert_array_equal(np.asarray(remapped)[0, 0, :6, 0], 0)
    assert np.asarray(remapped)[0, 1, 1, 0] == pytest.approx(np.log(2))
    with pytest.raises(ValueError, match="aligned"):
        preprocess_encoder_inputs(rgb, depth[:-1], 2)
