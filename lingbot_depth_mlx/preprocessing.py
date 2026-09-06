"""RGB-D loading and the exact v0.5 encoder input transform."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import mlx.core as mx
import mlx.nn as nn
import numpy as np

PATCH_SIZE = 14
IMAGE_MEAN = mx.array([0.485, 0.456, 0.406])
IMAGE_STD = mx.array([0.229, 0.224, 0.225])


def exact_scale(source: int, target: int) -> float:
    return math.nextafter(target / source, math.inf)


def load_rgb(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read RGB image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_depth(path: str | Path, unit: str = "mm") -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None or depth.ndim != 2:
        raise ValueError(f"cannot read single-channel depth: {path}")
    if unit not in {"mm", "m"}:
        raise ValueError(f"depth unit must be 'mm' or 'm', got {unit!r}")
    return depth.astype(np.float32) / (1000.0 if unit == "mm" else 1.0)


def token_grid(height: int, width: int, num_tokens: int) -> tuple[int, int]:
    if height <= 0 or width <= 0 or num_tokens <= 0:
        raise ValueError("height, width, and num_tokens must be positive")
    aspect_ratio = width / height
    return round((num_tokens / aspect_ratio) ** 0.5), round((num_tokens * aspect_ratio) ** 0.5)


def preprocess_encoder_inputs(
    rgb: np.ndarray, depth_m: np.ndarray, num_tokens: int
) -> tuple[mx.array, mx.array, mx.array, tuple[int, int]]:
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("RGB must be uint8 [H,W,3]")
    if depth_m.ndim != 2 or depth_m.shape != rgb.shape[:2]:
        raise ValueError("depth must be [H,W] and aligned with RGB")

    rows, cols = token_grid(*depth_m.shape, num_tokens)
    target_h, target_w = rows * PATCH_SIZE, cols * PATCH_SIZE
    scale = (exact_scale(depth_m.shape[0], target_h), exact_scale(depth_m.shape[1], target_w))
    image = nn.Upsample(scale, mode="linear", align_corners=False, antialias=True)(
        mx.array(rgb, dtype=mx.float32)[None] / 255.0
    )
    image = (image - IMAGE_MEAN) / IMAGE_STD
    # PyTorch legacy nearest uses floor(dst * src/dst_size), also when enlarging.
    # MLX Upsample uses half-pixel rounding for enlargement, which shifts holes.
    y = (mx.arange(target_h, dtype=mx.float32) * (depth_m.shape[0] / target_h)).astype(mx.int32)
    x = (mx.arange(target_w, dtype=mx.float32) * (depth_m.shape[1] / target_w)).astype(mx.int32)
    depth = mx.array(depth_m, dtype=mx.float32)[y[:, None], x[None, :]][None, ..., None]
    valid = mx.isfinite(depth) & (depth > 0.01)
    depth = mx.where(valid, mx.log(mx.where(valid, depth, mx.ones_like(depth))), 0.0)
    return image, depth, valid, (rows, cols)
