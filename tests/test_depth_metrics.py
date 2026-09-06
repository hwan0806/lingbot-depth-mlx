import numpy as np
import pytest

from metrics import depth_metrics, mask_iou


def test_exact_depth_metrics():
    depth = np.array([[1, 2], [3, 4]], np.float32)
    values = depth_metrics(depth, depth)
    assert values["rmse"] == values["mae"] == values["absrel"] == 0
    assert values["d102"] == values["d105"] == values["delta1"] == 1


def test_known_relative_error_and_invalid_exclusion():
    reference = np.array([[1, 2], [0, np.nan]], np.float32)
    prediction = np.array([[1.1, 2.2], [99, 99]], np.float32)
    values = depth_metrics(reference, prediction)
    assert values["absrel"] == pytest.approx(0.1)
    assert values["valid_coverage"] == 0.5


def test_mask_iou():
    assert mask_iou([[1, 1], [0, 0]], [[1, 0], [1, 0]]) == pytest.approx(1 / 3)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, 0, -1])
def test_invalid_prediction_cannot_improve_score_by_exclusion(bad):
    with pytest.raises(ValueError, match="invalid predictions at 1"):
        depth_metrics([[1, 2]], [[1, bad]])
    assert depth_metrics([[1, 2]], [[1, bad]], [[True, False]])["rmse"] == 0


def test_mask_shape_cannot_broadcast():
    with pytest.raises(ValueError, match="valid mask shape mismatch"):
        depth_metrics(np.ones((2, 2)), np.ones((2, 2)), [True, False])


def test_empty_valid_and_single_pixel():
    with pytest.raises(ValueError, match="no valid"):
        depth_metrics([[0]], [[1]])
    values = depth_metrics([[2]], [[2.2]])
    assert values["absrel"] == pytest.approx(0.1)
    assert values["boundary_coverage"] == 0


def test_invalid_gt_does_not_create_boundary():
    values = depth_metrics([[1, 1, 0, 1, 1]], [[1, 1, 99, 1, 1]])
    assert values["boundary_coverage"] == 0


def test_known_scale_and_large_finite_error():
    values = depth_metrics([[1, 2]], [[2, 4]])
    assert values["rmse"] == pytest.approx(np.sqrt(2.5))
    assert values["absrel"] == 1
    assert values["delta1"] == 0
    assert np.isfinite(depth_metrics([[1]], [[1e30]])["rmse"])


@pytest.mark.parametrize("delta", [-1, np.nan, np.inf])
def test_invalid_boundary_threshold(delta):
    with pytest.raises(ValueError, match="boundary_delta"):
        depth_metrics([[1]], [[1]], boundary_delta=delta)
