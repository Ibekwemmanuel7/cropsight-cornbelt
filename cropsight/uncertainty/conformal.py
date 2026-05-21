"""
cropsight.uncertainty.conformal
================================

Split conformal prediction for regression intervals.

Replaces the residual-bootstrap intervals used in the original module 3.
Split conformal gives a *distribution-free* marginal coverage guarantee:
under exchangeability of (X, y) across the calibration and test sets, the
returned interval covers the true y with probability >= 1 - alpha.

Why this matters for CropSight
------------------------------
Residual bootstrap assumes errors are i.i.d. - typically false across years
in agricultural forecasting (drought years have different error distributions
than trend years). Split conformal gives a proper coverage bound that an
insurance / lender Model Risk Management review can actually accept.

Year-based split for the in-season pipeline:
    train     : 2000 - 2019  (model fitting)
    calibrate : 2020 - 2021  (compute non-conformity scores)
    val       : 2022          (early stopping / hyperparam tuning if any)
    test      : 2023          (final evaluation)

Public API
----------
split_conformal_quantile(residuals, alpha) -> float
predict_with_interval(model, X, q_alpha) -> (pred, lower, upper)
evaluate_coverage(y_true, lower, upper) -> float
evaluate_mean_width(lower, upper) -> float
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def split_conformal_quantile(residuals: np.ndarray, alpha: float = 0.1) -> float:
    """
    Compute the conformal quantile from a calibration set of absolute residuals.

    For a 1-alpha = 90% interval, returns the (1-alpha) quantile of |residuals|
    with the conformal correction: index = ceil((n+1) * (1-alpha)) - 1.

    Parameters
    ----------
    residuals : 1-D array of absolute residuals |y - y_pred| on the calibration set
    alpha     : miscoverage rate (default 0.1 for 90% intervals)

    Returns
    -------
    q_alpha : float - the interval half-width
    """
    residuals = np.asarray(residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    n = residuals.size
    if n == 0:
        raise ValueError("split_conformal_quantile: empty residuals")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")

    # Conformal correction: rank = ceil((n+1)*(1-alpha))
    # Clamp to n to avoid index overflow on small calibration sets.
    rank = int(math.ceil((n + 1) * (1.0 - alpha)))
    rank = min(rank, n)
    sorted_resid = np.sort(residuals)
    return float(sorted_resid[rank - 1])


def predict_with_interval(
    model: Any,
    X: np.ndarray,
    q_alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Produce point predictions and a symmetric conformal interval [pred - q, pred + q].

    Returns
    -------
    pred  : array of point predictions
    lower : pred - q_alpha
    upper : pred + q_alpha
    """
    pred = np.asarray(model.predict(X), dtype=float)
    lower = pred - q_alpha
    upper = pred + q_alpha
    return pred, lower, upper


def evaluate_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Empirical coverage = fraction of y_true inside [lower, upper]."""
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside))


def evaluate_mean_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """Mean width of the prediction interval (upper - lower)."""
    return float(np.mean(np.asarray(upper) - np.asarray(lower)))


def calibrate_and_predict(
    model: Any,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    X_test: np.ndarray,
    alpha: float = 0.1,
) -> dict[str, Any]:
    """
    One-shot calibration + test prediction.

    Returns a dict:
        q_alpha    : conformal quantile from calibration residuals
        pred       : test point predictions
        lower      : test interval lower bound
        upper      : test interval upper bound
        calib_n    : number of calibration samples used
    """
    calib_pred = np.asarray(model.predict(X_calib), dtype=float)
    calib_resid = np.abs(np.asarray(y_calib, dtype=float) - calib_pred)
    q_alpha = split_conformal_quantile(calib_resid, alpha=alpha)
    pred, lower, upper = predict_with_interval(model, X_test, q_alpha)
    return {
        "q_alpha": q_alpha,
        "pred": pred,
        "lower": lower,
        "upper": upper,
        "calib_n": int(np.isfinite(calib_resid).sum()),
    }
