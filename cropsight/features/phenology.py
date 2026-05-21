"""
cropsight.features.phenology
============================

Phenology features for in-season corn yield forecasting.

The "as of week K" rule: no input data point used to derive any feature may have
a DOY greater than K * 7. Phenology features are NDVI-derived. NDVI input is
MOD13Q1 (16-day composite); at week K only observations with doy <= K*7 are
visible.

Features that require post-cutoff information return NaN by design (e.g.
`eos_doy` is NaN until late October). Features that are partially observed
return the mean over the observed portion (e.g. `ndvi_silking` mid-silking
returns the mean over the elapsed days of the silking window).

Public API
----------
smooth_truncated(doys, ndvi, week_k, doy_grid=None) -> np.ndarray
extract_phenology_to_week(ndvi_smooth, week_k, doy_grid=None) -> dict
compute_vci_to_week(ndvi_df, week_k, baseline_years) -> pd.DataFrame
vci_window_means_to_week(vci_df, week_k) -> pd.DataFrame

See docs/specs/in_season_pipeline.md for the full spec.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import UnivariateSpline

DEFAULT_DOY_GRID = np.arange(60, 331)  # early March through late November

# Phenological windows (DOY ranges) for US Corn Belt corn
WINDOWS: dict[str, tuple[int, int]] = {
    "vegetative": (130, 180),
    "silking": (180, 220),
    "grainfill": (220, 270),
}

PHENOLOGY_FEATURE_KEYS: list[str] = [
    "sos_doy",
    "eos_doy",
    "peak_ndvi",
    "peak_doy",
    "season_length",
    "integrated_ndvi",
    "greenup_rate",
    "senescence_rate",
    "ndvi_vegetative",
    "ndvi_silking",
    "ndvi_grainfill",
]


def smooth_truncated(
    doys: np.ndarray,
    ndvi: np.ndarray,
    week_k: int,
    doy_grid: np.ndarray | None = None,
) -> np.ndarray:
    """
    Spline + Savitzky-Golay smoothing of NDVI, restricted to observations with
    doy <= week_k * 7. The returned array has the same shape as `doy_grid`, with
    NaN entries for any grid DOY beyond the cutoff. This means the caller does
    not need to re-apply the cutoff after smoothing.

    Returns NaN-only if fewer than 4 valid observations exist within the cutoff.
    """
    if doy_grid is None:
        doy_grid = DEFAULT_DOY_GRID

    cutoff_doy = week_k * 7
    doys = np.asarray(doys, dtype=float)
    ndvi = np.asarray(ndvi, dtype=float)

    valid = (~np.isnan(ndvi)) & (doys <= cutoff_doy)
    if int(valid.sum()) < 4:
        return np.full(len(doy_grid), np.nan)

    d_in = doys[valid]
    v_in = ndvi[valid]
    order = np.argsort(d_in)
    d_in, v_in = d_in[order], v_in[order]

    in_window = doy_grid <= cutoff_doy
    fitted = np.full(len(doy_grid), np.nan)

    try:
        spl = UnivariateSpline(d_in, v_in, k=3, s=0.02)
        fitted_window = np.clip(spl(doy_grid[in_window]), -0.1, 1.0)
    except Exception:
        fitted_window = np.interp(doy_grid[in_window], d_in, v_in)

    if fitted_window.size > 21:
        fitted_window = np.clip(signal.savgol_filter(fitted_window, 21, 3), -0.1, 1.0)

    fitted[in_window] = fitted_window
    return fitted


def extract_phenology_to_week(
    ndvi_smooth: np.ndarray,
    week_k: int,
    doy_grid: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Extract phenology features from a truncated, smoothed NDVI series.

    The series must already have NaN entries for doy > week_k*7 (this is what
    `smooth_truncated` produces). Features unobservable at week_k return NaN.

    Returns a dict with all keys in PHENOLOGY_FEATURE_KEYS, always.
    """
    if doy_grid is None:
        doy_grid = DEFAULT_DOY_GRID

    cutoff_doy = week_k * 7
    out: dict[str, float] = {k: np.nan for k in PHENOLOGY_FEATURE_KEYS}

    finite = np.isfinite(ndvi_smooth) & (doy_grid <= cutoff_doy)
    if int(finite.sum()) < 4:
        return out

    obs_vals = ndvi_smooth[finite]
    obs_doys = doy_grid[finite].astype(float)

    base = float(np.nanpercentile(obs_vals, 10))
    peak_val = float(np.nanmax(obs_vals))
    peak_idx = int(np.nanargmax(obs_vals))
    peak_doy = float(obs_doys[peak_idx])
    threshold = base + 0.5 * (peak_val - base)
    above_threshold = obs_vals >= threshold

    # SOS = first DOY >= 100 where smoothed NDVI exceeds threshold
    sos_idx = None
    for i, (dy, ab) in enumerate(zip(obs_doys, above_threshold, strict=False)):
        if dy >= 100 and ab:
            sos_idx = i
            break

    if sos_idx is not None:
        out["sos_doy"] = float(obs_doys[sos_idx])

    out["peak_ndvi"] = peak_val
    out["peak_doy"] = peak_doy

    # EOS only knowable once cutoff has reached late October (DOY 287 = week 41)
    eos_doy: float = float("nan")
    if cutoff_doy >= 287:
        eos_idx = None
        for i in range(len(obs_vals) - 1, -1, -1):
            if obs_doys[i] <= 310 and above_threshold[i]:
                eos_idx = i
                break
        if eos_idx is not None and eos_idx < len(obs_vals) - 1:
            eos_doy = float(obs_doys[eos_idx])
    out["eos_doy"] = eos_doy

    # Integrated NDVI: trapezoidal over [sos, cutoff] in DOY space
    if sos_idx is not None:
        seg_vals = obs_vals[sos_idx:]
        seg_doys = obs_doys[sos_idx:]
        if seg_doys.size >= 2:
            try:
                out["integrated_ndvi"] = float(np.trapezoid(seg_vals, seg_doys))
            except AttributeError:
                out["integrated_ndvi"] = float(np.trapz(seg_vals, seg_doys))

    # Greenup rate: slope from SOS to peak-so-far
    if sos_idx is not None and peak_doy > obs_doys[sos_idx]:
        out["greenup_rate"] = float(
            (peak_val - obs_vals[sos_idx]) / max(peak_doy - obs_doys[sos_idx], 1.0)
        )

    # Senescence rate: requires at least 2 post-peak observations
    post_peak_count = int((obs_doys > peak_doy).sum())
    if not np.isnan(eos_doy) and peak_idx < len(obs_vals) - 2 and post_peak_count >= 2:
        out["senescence_rate"] = float(
            (obs_vals[-1] - peak_val) / max(obs_doys[-1] - peak_doy, 1.0)
        )

    if not np.isnan(out["sos_doy"]) and not np.isnan(out["eos_doy"]):
        out["season_length"] = float(out["eos_doy"] - out["sos_doy"])

    # Window means - gate by whether window has started, partial if mid-window
    for name, (window_start, window_end) in WINDOWS.items():
        if cutoff_doy < window_start:
            continue  # leave NaN
        window_end_effective = min(window_end, cutoff_doy)
        mask = (obs_doys >= window_start) & (obs_doys <= window_end_effective)
        if mask.any():
            out[f"ndvi_{name}"] = float(np.nanmean(obs_vals[mask]))

    return out


def compute_vci_to_week(
    ndvi_df: pd.DataFrame,
    week_k: int,
    baseline_years: Iterable[int],
) -> pd.DataFrame:
    """
    Vegetation Condition Index (VCI) normalized against per-(fips, doy_bin)
    historical extremes computed over `baseline_years` only.

    To avoid leakage:
      - baseline rows are restricted to baseline_years
      - all rows are restricted to doy <= week_k*7

    The caller is responsible for choosing baseline_years that exclude the
    forecast year and any future years.

    Input columns required: fips, year, doy, ndvi_mean
    Output columns: fips, year, doy, vci
    """
    cutoff_doy = week_k * 7
    baseline_years = set(int(y) for y in baseline_years)

    df = ndvi_df.copy()
    df["doy_bin"] = ((df["doy"] // 16) * 16).astype("int64")
    df = df[df["doy"] <= cutoff_doy].copy()

    base = df[df["year"].isin(baseline_years)]
    hist = (
        base.groupby(["fips", "doy_bin"])["ndvi_mean"]
        .agg(ndvi_hist_min="min", ndvi_hist_max="max")
        .reset_index()
    )
    hist["doy_bin"] = hist["doy_bin"].astype("int64")

    merged = df.merge(hist, on=["fips", "doy_bin"], how="left")
    denom = merged["ndvi_hist_max"] - merged["ndvi_hist_min"]
    vci = np.where(
        denom > 0.01,
        100.0 * (merged["ndvi_mean"] - merged["ndvi_hist_min"]) / denom,
        50.0,
    )
    merged["vci"] = np.clip(vci, 0.0, 100.0)
    return merged[["fips", "year", "doy", "vci"]]


def vci_window_means_to_week(
    vci_df: pd.DataFrame,
    week_k: int,
) -> pd.DataFrame:
    """
    Per (fips, year), compute window-mean VCI for vegetative, silking,
    grainfill, and full-season, restricted to doy <= week_k*7.

    If a window has not started by week K, that column is omitted (the merge
    will produce NaN).

    Input columns required: fips, year, doy, vci
    Output columns: fips, year, vci_vegetative?, vci_silking?, vci_grainfill?, vci_season?
    """
    cutoff_doy = week_k * 7
    df = vci_df[vci_df["doy"] <= cutoff_doy]

    frames: list[pd.DataFrame] = []
    for name, (window_start, window_end) in WINDOWS.items():
        if cutoff_doy < window_start:
            continue
        window_end_effective = min(window_end, cutoff_doy)
        sub = df[(df["doy"] >= window_start) & (df["doy"] <= window_end_effective)]
        if sub.empty:
            continue
        agg = (
            sub.groupby(["fips", "year"])["vci"]
            .mean()
            .reset_index()
            .rename(columns={"vci": f"vci_{name}"})
        )
        frames.append(agg)

    sub_season = df[(df["doy"] >= 100) & (df["doy"] <= cutoff_doy)]
    if not sub_season.empty:
        agg_season = (
            sub_season.groupby(["fips", "year"])["vci"]
            .mean()
            .reset_index()
            .rename(columns={"vci": "vci_season"})
        )
        frames.append(agg_season)

    if not frames:
        return pd.DataFrame(columns=["fips", "year"])

    result = frames[0]
    for f in frames[1:]:
        result = result.merge(f, on=["fips", "year"], how="outer")
    return result
