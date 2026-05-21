"""
Tests for cropsight.features.phenology.

The goal is to catch leakage of post-cutoff data into features. Each test
builds a synthetic NDVI series with known structure, smooths it at a chosen
week K, and asserts properties of the output.
"""

from __future__ import annotations

import numpy as np
import pytest

from cropsight.features import phenology


def _synthetic_season(amp: float = 0.8, peak_doy: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Bell-curve NDVI peaking at peak_doy with amplitude `amp`. 16-day cadence."""
    doys = np.arange(60, 331, 16)
    width = 60.0
    ndvi = 0.15 + amp * np.exp(-(((doys - peak_doy) / width) ** 2))
    return doys, ndvi


def test_smooth_truncated_respects_cutoff():
    """Observations past the cutoff must not influence the smooth."""
    doys, ndvi = _synthetic_season(amp=0.8, peak_doy=200)
    smooth_k28 = phenology.smooth_truncated(doys, ndvi, week_k=28)  # cutoff DOY 196

    grid = phenology.DEFAULT_DOY_GRID
    assert np.all(np.isnan(smooth_k28[grid > 196])), "smooth_truncated leaked into doy > cutoff"
    # And there should be valid output up to the cutoff
    assert np.isfinite(smooth_k28[grid <= 196]).sum() > 50


def test_smooth_truncated_too_few_points():
    """If fewer than 4 observations within cutoff, output is all NaN."""
    doys = np.array([60, 76, 92])  # only 3 points, before any K
    ndvi = np.array([0.2, 0.3, 0.4])
    out = phenology.smooth_truncated(doys, ndvi, week_k=20)
    assert np.all(np.isnan(out))


def test_extract_phenology_gates_at_k20():
    """At K=20 (DOY 140) silking/grainfill/EOS features must be NaN."""
    doys, ndvi = _synthetic_season(amp=0.8, peak_doy=200)
    smooth = phenology.smooth_truncated(doys, ndvi, week_k=20)
    feats = phenology.extract_phenology_to_week(smooth, week_k=20)

    assert np.isnan(feats["ndvi_silking"]), "ndvi_silking leaked at K=20"
    assert np.isnan(feats["ndvi_grainfill"]), "ndvi_grainfill leaked at K=20"
    assert np.isnan(feats["eos_doy"]), "eos_doy leaked at K=20"
    assert np.isnan(feats["senescence_rate"]), "senescence_rate leaked at K=20"
    assert np.isnan(feats["season_length"]), "season_length leaked at K=20"


def test_extract_phenology_at_k32_silking_observed():
    """At K=32 (DOY 224) the silking window has fully closed (180-220)."""
    doys, ndvi = _synthetic_season(amp=0.8, peak_doy=200)
    smooth = phenology.smooth_truncated(doys, ndvi, week_k=32)
    feats = phenology.extract_phenology_to_week(smooth, week_k=32)

    assert np.isfinite(feats["ndvi_silking"]), "ndvi_silking should be observed at K=32"
    assert feats["ndvi_silking"] > 0.5, "ndvi_silking too low for a peak-around-200 season"
    # Grainfill window starts at 220, cutoff is 224 -> only 4 days of partial overlap
    # Should be partial; sometimes NaN depending on the 16-day cadence
    # Either NaN or a real value, but not garbage
    if np.isfinite(feats["ndvi_grainfill"]):
        assert 0.0 <= feats["ndvi_grainfill"] <= 1.0


def test_peak_doy_is_capped_at_cutoff():
    """peak_doy cannot exceed cutoff_doy = week_k * 7."""
    doys, ndvi = _synthetic_season(amp=0.9, peak_doy=210)  # true peak at 210
    for k in (20, 24, 28, 32):
        smooth = phenology.smooth_truncated(doys, ndvi, week_k=k)
        feats = phenology.extract_phenology_to_week(smooth, week_k=k)
        cutoff = k * 7
        if np.isfinite(feats["peak_doy"]):
            assert feats["peak_doy"] <= cutoff, (
                f"K={k}: peak_doy={feats['peak_doy']} > cutoff {cutoff}"
            )


def test_extract_phenology_always_returns_all_keys():
    """The output dict must always contain every key in PHENOLOGY_FEATURE_KEYS."""
    doys = np.array([60, 76, 92, 108])
    ndvi = np.array([0.2, 0.25, 0.3, 0.35])
    smooth = phenology.smooth_truncated(doys, ndvi, week_k=20)
    feats = phenology.extract_phenology_to_week(smooth, week_k=20)
    assert set(feats.keys()) == set(phenology.PHENOLOGY_FEATURE_KEYS)


def test_eos_doy_only_observable_after_doy_287():
    """EOS feature must be NaN for K < 41."""
    doys, ndvi = _synthetic_season(amp=0.8, peak_doy=200)
    for k in (20, 24, 28, 32, 36, 40):
        smooth = phenology.smooth_truncated(doys, ndvi, week_k=k)
        feats = phenology.extract_phenology_to_week(smooth, week_k=k)
        assert np.isnan(feats["eos_doy"]), f"eos_doy leaked at K={k}"
