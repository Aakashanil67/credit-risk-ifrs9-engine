import numpy as np
import pytest

from src.calibration import cross_fitted_calibration, fit_calibrator


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_calibrators_return_bounded_probabilities(method):
    y = np.array([0, 1] * 50)
    raw = np.linspace(0.01, 0.99, 100)

    calibrator = fit_calibrator(method, y, raw)
    calibrated = calibrator.predict(np.array([0.0, 0.2, 0.8, 1.0]))

    assert np.isfinite(calibrated).all()
    assert ((0 <= calibrated) & (calibrated <= 1)).all()


def test_cross_fitted_calibration_scores_every_row_once_and_is_deterministic():
    y = np.array([0, 1] * 50)
    raw = np.tile(np.array([0.15, 0.35, 0.55, 0.75]), 25)

    first = cross_fitted_calibration(y, raw, method="sigmoid", folds=5, seed=42)
    second = cross_fitted_calibration(y, raw, method="sigmoid", folds=5, seed=42)

    np.testing.assert_allclose(first, second)
    assert first.shape == raw.shape
    assert np.isfinite(first).all()


def test_unknown_calibration_method_is_rejected():
    with pytest.raises(ValueError, match="sigmoid or isotonic"):
        fit_calibrator("spline", np.array([0, 1]), np.array([0.2, 0.8]))
