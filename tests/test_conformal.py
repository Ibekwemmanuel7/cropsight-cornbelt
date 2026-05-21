"""
Tests for cropsight.uncertainty.conformal.

The key property we verify: empirical coverage on held-out test points
matches the nominal 1 - alpha across multiple random calibration/test
splits. Conformal prediction is distribution-free under exchangeability;
these tests use i.i.d. samples so exchangeability holds by construction.
"""

from __future__ import annotations

import numpy as np
import pytest

from cropsight.uncertainty import conformal


def test_quantile_basic():
    """Hand-checked small example: residuals sorted, picked at rank ceil((n+1)*0.9)."""
    # n=10, alpha=0.1 -> rank = ceil(11 * 0.9) = ceil(9.9) = 10
    # So q = sorted_resid[9] = max
    residuals = np.array([0.1, 0.2, 0.5, 0.3, 0.4, 0.7, 0.6, 0.9, 0.8, 1.0])
    q = conformal.split_conformal_quantile(residuals, alpha=0.1)
    assert q == 1.0

    # alpha=0.2 -> rank = ceil(11 * 0.8) = 9 -> sorted_resid[8] = 0.9
    q = conformal.split_conformal_quantile(residuals, alpha=0.2)
    assert q == 0.9


def test_quantile_rejects_bad_alpha():
    with pytest.raises(ValueError):
        conformal.split_conformal_quantile(np.array([1.0, 2.0]), alpha=0.0)
    with pytest.raises(ValueError):
        conformal.split_conformal_quantile(np.array([1.0, 2.0]), alpha=1.0)
    with pytest.raises(ValueError):
        conformal.split_conformal_quantile(np.array([1.0, 2.0]), alpha=-0.1)


def test_quantile_rejects_empty():
    with pytest.raises(ValueError):
        conformal.split_conformal_quantile(np.array([]), alpha=0.1)


def test_quantile_filters_nan():
    """NaN residuals should be silently filtered out."""
    residuals = np.array([0.5, np.nan, 1.0, np.nan, 2.0])
    q = conformal.split_conformal_quantile(residuals, alpha=0.1)
    # After filtering NaN, n=3, rank=ceil(4*0.9)=4, clamped to 3 -> max
    assert q == 2.0


def test_coverage_matches_nominal_on_iid_data():
    """
    Average empirical coverage across multiple random splits should be close
    to 1 - alpha. Use moderately large samples to keep variance small.
    """
    rng_master = np.random.default_rng(42)
    target = 0.90
    covs = []
    for _trial in range(30):
        seed = int(rng_master.integers(1_000_000))
        rng = np.random.default_rng(seed)
        # Calibration absolute residuals from N(0, 1)
        calib_resid = np.abs(rng.normal(0, 1, 2000))
        q = conformal.split_conformal_quantile(calib_resid, alpha=0.1)
        # Test "true" values from same N(0, 1); model predicts 0
        y = rng.normal(0, 1, 3000)
        cov = conformal.evaluate_coverage(y, -q * np.ones_like(y), q * np.ones_like(y))
        covs.append(cov)
    mean_cov = float(np.mean(covs))
    # With 30 trials of 3000 test points and n_calib=2000, the standard error
    # on the mean is roughly 0.001-0.003. Allow generous tolerance for CI stability.
    assert abs(mean_cov - target) < 0.02, (
        f"mean conformal coverage {mean_cov:.4f} not close to {target}"
    )


def test_evaluate_coverage_simple():
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    lower = np.array([5.0, 25.0, 25.0, 35.0])
    upper = np.array([15.0, 35.0, 35.0, 45.0])
    # 10 in [5,15] yes; 20 in [25,35] no; 30 in [25,35] yes; 40 in [35,45] yes
    assert conformal.evaluate_coverage(y_true, lower, upper) == 0.75


def test_evaluate_mean_width():
    lower = np.array([0.0, 10.0, 20.0])
    upper = np.array([2.0, 14.0, 26.0])
    # widths 2, 4, 6 -> mean 4
    assert conformal.evaluate_mean_width(lower, upper) == 4.0


def test_calibrate_and_predict_endtoend():
    """Mock model wrapper to check the end-to-end path."""

    class _FakeModel:
        def __init__(self, slope: float = 1.0, intercept: float = 0.0):
            self.slope, self.intercept = slope, intercept

        def predict(self, X: np.ndarray) -> np.ndarray:
            return self.slope * X[:, 0] + self.intercept

    rng = np.random.default_rng(0)
    X_calib = rng.normal(size=(500, 1))
    y_calib = X_calib[:, 0] + rng.normal(0, 1, 500)  # truth = X + noise
    X_test = rng.normal(size=(2000, 1))
    y_test = X_test[:, 0] + rng.normal(0, 1, 2000)

    model = _FakeModel(slope=1.0)
    result = conformal.calibrate_and_predict(model, X_calib, y_calib, X_test, alpha=0.1)
    assert result["calib_n"] == 500
    assert result["pred"].shape == (2000,)
    assert result["lower"].shape == (2000,)
    assert result["upper"].shape == (2000,)

    cov = conformal.evaluate_coverage(y_test, result["lower"], result["upper"])
    assert 0.85 < cov < 0.95, f"end-to-end coverage off target: {cov}"
