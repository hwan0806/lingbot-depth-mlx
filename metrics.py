"""Depth and mask regression metrics used by the CLI and harnesses."""

from __future__ import annotations

import numpy as np


def depth_metrics(reference, prediction, valid=None, boundary_delta: float = 0.05) -> dict[str, float]:
    reference = np.asarray(reference, np.float64)
    prediction = np.asarray(prediction, np.float64)
    if reference.shape != prediction.shape:
        raise ValueError(f"shape mismatch: {reference.shape} != {prediction.shape}")
    if reference.ndim < 2:
        raise ValueError("depth must have at least two spatial dimensions")
    if not np.isfinite(boundary_delta) or boundary_delta < 0:
        raise ValueError("boundary_delta must be finite and nonnegative")
    finite = np.isfinite(reference) & (reference > 0)
    if valid is not None:
        valid = np.asarray(valid, bool)
        if valid.shape != reference.shape:
            raise ValueError(f"valid mask shape mismatch: {valid.shape} != {reference.shape}")
    valid = finite if valid is None else finite & valid
    if not valid.any():
        raise ValueError("no valid depth pixels")
    invalid_predictions = valid & (~np.isfinite(prediction) | (prediction <= 0))
    if invalid_predictions.any():
        raise ValueError(f"invalid predictions at {int(invalid_predictions.sum())} valid depth pixels")
    error = prediction[valid] - reference[valid]
    absolute = np.abs(error)
    ratio = np.maximum(prediction[valid] / reference[valid], reference[valid] / np.maximum(prediction[valid], 1e-8))
    # Invalid or excluded GT neighbors must not manufacture depth boundaries.
    safe_reference = np.where(valid, reference, 0)
    gradients = []
    boundary_valid = valid.copy()
    for axis in (-2, -1):
        if reference.shape[axis] == 1:
            gradients.append(np.zeros_like(reference))
            continue
        gradients.append(np.gradient(safe_reference, axis=axis))
        for shift in (-1, 1):
            neighbor_valid = np.roll(valid, shift, axis=axis)
            edge = [slice(None)] * reference.ndim
            edge[axis] = 0 if shift == 1 else -1
            neighbor_valid[tuple(edge)] = valid[tuple(edge)]
            boundary_valid &= neighbor_valid
    boundary = boundary_valid & (np.hypot(*gradients) >= boundary_delta)
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(absolute.mean()),
        "absrel": float(np.mean(absolute / reference[valid])),
        "d102": float(np.mean(ratio < 1.02)),
        "d105": float(np.mean(ratio < 1.05)),
        "delta1": float(np.mean(ratio < 1.25)),
        "valid_coverage": float(valid.mean()),
        "boundary_coverage": float(boundary.mean()),
        "boundary_rmse": float(np.sqrt(np.mean((prediction[boundary] - reference[boundary]) ** 2))) if boundary.any() else 0.0,
        "max_abs_error": float(absolute.max()),
    }


def mask_iou(reference, prediction) -> float:
    reference, prediction = np.asarray(reference, bool), np.asarray(prediction, bool)
    if reference.shape != prediction.shape:
        raise ValueError(f"shape mismatch: {reference.shape} != {prediction.shape}")
    union = np.logical_or(reference, prediction).sum()
    return float(np.logical_and(reference, prediction).sum() / union) if union else 1.0
