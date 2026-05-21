"""
cropsight.uncertainty
=====================

Prediction-interval and calibration utilities.

Currently exports:
    conformal.split_conformal_quantile
    conformal.predict_with_interval
    conformal.evaluate_coverage
    conformal.evaluate_mean_width
"""

from . import conformal

__all__ = ["conformal"]
