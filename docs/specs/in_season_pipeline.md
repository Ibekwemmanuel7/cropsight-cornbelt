# In-season feature pipeline — technical spec

**Owner:** Emmanuel · **Status:** draft for sign-off · **Updated:** 2026-05-20

## Goal

Transform CropSight from a hindcast (forecast using full-season features, validated against the year just ended) into an in-season product (forecast using only features observable up to forecast week K, refreshed weekly during the growing season). This is the technical work that turns "research repo" into "thing customers pay for."

After this build, the product can answer **"what will county FIPS X yield, given everything observable through today (week K)?"** at K = 20, 24, 28, 32, 36 — corresponding to mid-May through early September.

## Why now

The strategy doc identifies this as the single highest-leverage piece of engineering. Every customer conversation (agronomist client meetings, insurance underwriting, ag lender risk pricing) cares about the question "what is the season going to do?" *before* the season ends. A hindcast is a portfolio piece; in-season forecasting is a product.

## Forecast horizons

We will produce forecasts at six week-K cuts, defined as cumulative-NDVI-day-of-year `≤ K * 7`:

| K | DOY cutoff | Calendar (typical) | Crop phase | Expected use |
|---|---|---|---|---|
| 16 | 112 | mid-April | pre-planting | climatology baseline (low value) |
| 20 | 140 | mid-May | planting / emergence | first in-season forecast |
| 24 | 168 | mid-June | vegetative | growers planning side-dress N |
| 28 | 196 | mid-July | pre-silking | the high-stakes window — heat/drought risk |
| 32 | 224 | mid-August | silking → early grain fill | best accuracy window |
| 36 | 252 | early September | grain fill | near-final, used for marketing decisions |

Week 16 stays in the pipeline as a control but is not marketed as a product tier — at that point the model is climatology + soil + year_trend, which any extension agronomist already knows.

## Feature trimming rules

For each feature, define how it is computed at week K. **No input data point used to derive any K-feature may have a DOY greater than `K * 7`.** This is the leakage rule.

### Phenology (15 features)

NDVI is MOD13Q1 — a 16-day composite. As of week K we have whichever composites have a `system:time_start` DOY ≤ `K * 7`. For K=20 (DOY 140) that's roughly the DOY-60, 76, 92, 108, 124 composites — five observations. For K=32 (DOY 224) it's twelve observations. The Savitzky-Golay smoother in `smooth_ndvi_series` should be re-fit on the truncated series, not on the full season.

| Feature | At-week-K rule |
|---|---|
| `sos_doy` | Compute only if SOS already detected by DOY ≤ K·7; else NaN |
| `eos_doy` | NaN until K ≥ 41 (DOY ≥ 287) |
| `peak_ndvi` | `max(ndvi_smooth_truncated)` — note this is "peak so far," not final peak |
| `peak_doy` | argmax of truncated smooth |
| `season_length` | NaN until both SOS and EOS detected |
| `integrated_ndvi` | `trapz` over `[sos_doy, K·7]`, not over full season |
| `greenup_rate` | Compute if peak_so_far passed; else "growth slope from SOS to current DOY" |
| `senescence_rate` | NaN until at least 2 post-peak observations exist |
| `ndvi_vegetative` (DOY 130-180) | Mean if K·7 ≥ 180; partial mean if K·7 ∈ [130, 180]; NaN if K·7 < 130 |
| `ndvi_silking` (DOY 180-220) | Full if K·7 ≥ 220; partial if in window; NaN if before |
| `ndvi_grainfill` (DOY 220-270) | Full if K·7 ≥ 270; partial if in window; NaN if before |
| `vci_*` | Same gating as the matching `ndvi_*` |

**VCI baseline leakage:** the current `compute_vci` function builds historical min/max **using all years' NDVI** at each `doy_bin`. That's fine when training on past years, but the per-county min/max for a given DOY must be computed using only the training-window years. We already split train/val/test by year, so this is preserved as long as we re-fit the VCI baseline inside the K-cut training routine.

### Weather (11 features) — ERA5 path

Once `scripts/download_era5.py` finishes, we have daily T2m, Td2m, total precip, SSRD per grid cell. Aggregate to county centroids (already scaffolded in module 1 cell `b1000013`). For each (fips, year, K):

| Feature | At-week-K rule |
|---|---|
| `gdd_season` | Sum of daily GDD from Jan 1 (or DOY 90) through DOY = K·7 |
| `gdd_silking` (DOY 180-220) | Only valid if K·7 ≥ 220; partial if in window |
| `gdd_grainfill` (DOY 220-270) | Same gating |
| `precip_season_mm` | Cumulative precip through DOY = K·7 |
| `precip_silking_mm` | Gated as above |
| `heat_stress_days` | Count of days with Tmax > 35°C through K·7 |
| `spi_season` | Z-score of cumulative-to-K precip vs 2000–[K-1] cumulative-to-same-DOY precip |
| `vpd_season` | Mean VPD (from T2m and Td2m via Tetens) through K·7 |

**SPI is the trickiest:** baseline is the cumulative-to-same-DOY precip across historical years (not full-season historical). Naive SPI computed on full-season precip would leak future weather.

While ERA5 is downloading we should still ship a working in-season prototype against the existing proxy weather — see "phased rollout" below.

### DSSAT-proxy water balance (7 features)

The water balance in `run_water_balance` integrates daily over `n = 150` days. For week-K, run the balance over `n = K·7 - DOY_planting` days, ending at DOY = K·7. Inputs (precip, PET) must be truncated to the same window.

This means `water_stress_frac`, `aet_season_mm`, etc. all become "as of week K" quantities. The PINN's physics penalty still applies — increasing `water_stress_frac` should still raise predicted yield.

### Static features (7)

Soil (6) and `year_trend` (1) are time-invariant and require no trimming. They are the same value at every K.

## Architecture: where the in-season code lives

I propose adding a new module rather than rewriting module 2 in place. Module 2 stays as the full-season pipeline (still used for the historical leaderboard and the README hindcast story). The new module produces K-conditioned feature matrices.

```
cropsight-cornbelt/
├── module1_data_ingestion.ipynb            # unchanged
├── module2_feature_engineering.ipynb       # unchanged - full-season features
├── module2b_in_season_features.ipynb       # NEW - produces feature_matrix_K.parquet for each K
├── module3_modeling.ipynb                  # unchanged - full-season XGBoost+PINN
├── module3b_in_season_models.ipynb         # NEW - trains weekly XGBoost variants
└── cropsight/
    └── features/
        ├── __init__.py
        ├── phenology.py                    # NEW - smooth_truncated, extract_phenology_to_week
        ├── weather.py                      # NEW - gdd_to_week, precip_to_week, spi_to_week, vpd_to_week
        ├── water_balance.py                # NEW - run_water_balance_to_week
        └── leakage.py                      # NEW - audit_no_future_data(features, week_k)
```

The `cropsight/features/` Python package is new — extracting the logic from notebooks into testable modules. The notebooks import and call. This is a quality-of-life refactor we should have done anyway; in-season forces the issue.

## Outputs

For each `K ∈ {16, 20, 24, 28, 32, 36}` produce:

- `data/interim/feature_matrix_k{K}.parquet` — the full 2000-2023 county-year matrix evaluated at week K
- `data/interim/train_k{K}.parquet`, `val_k{K}.parquet`, `test_k{K}.parquet` — split by year (same 2000-2021/2022/2023 split)
- `models/xgboost_k{K}.json` — XGBoost trained on the K matrix
- `models/pinn_k{K}.pt` — PINN trained on the K matrix
- `data/interim/predictions_k{K}.parquet` — test-set predictions with 90% CIs (residual bootstrap; conformal later)

Plus one rollup:

- `data/interim/horizon_leaderboard.parquet` — one row per (K, model), columns RMSE/MAE/R²/bias on val and test. This becomes the headline chart.

## Validation methodology

Same temporal split as the current pipeline: train 2000-2021, val 2022, test 2023. No K-conditional split — every K-cut sees the same county-years, just with different feature availability.

The headline result is a chart: **"forecast accuracy as a function of week K."** Y-axis: test-set RMSE. X-axis: forecast week. Expect a monotonic decrease from K=16 (climatology floor) to K=36 (near hindcast).

We will also produce a second validation: a 2024 walk-forward backtest. For 2024, we don't yet have NASS ground truth (released ~Feb 2025), so this initially can't be evaluated against truth — but the prediction series itself (six weekly forecasts) is what we'll show agronomists as the live demo for the 2024 season. Once 2024 NASS lands we add an out-of-sample point to the leaderboard.

## Leakage audit checklist

Before any K-model trains, the audit module runs the following checks and fails loudly on violation. This is mandatory — no shortcuts.

1. **No-future-DOY check.** For each feature column derived from NDVI/weather/water-balance time series, assert no input row used in its computation has `doy > K*7`.
2. **VCI baseline check.** Min/max NDVI per (fips, doy_bin) used for VCI normalization must come exclusively from training-window years (2000-2021 when training, full record only at inference time on already-known years).
3. **SPI baseline check.** Cumulative-to-DOY precip baseline must be computed from years where that DOY has already passed at the time the SPI is being evaluated.
4. **Static feature check.** Soil and year_trend are constant within a (fips, year); assert no variance across K cuts.
5. **Cross-K column check.** Matrix `feature_matrix_k{K}.parquet` contains no columns that should be NaN at this K (e.g. `eos_doy` at K=24 must be NaN-only).
6. **Train/val/test temporal check.** Assert no overlap in years across the three splits.

The checklist lives in `cropsight/features/leakage.py` and is invoked from the K-cut notebook before each model fits.

## Phased rollout (4 phases, ~3 weeks)

**Phase 1 — Skeleton + leakage audit (3-4 days).** Stand up `cropsight/features/` package with `smooth_truncated`, `extract_phenology_to_week`, the leakage audit. Produce `feature_matrix_k28.parquet` end-to-end using the existing proxy weather (since ERA5 download is still running in background). Validate that K=28 features are sensible — peak NDVI roughly matches true peak by DOY 196.

**Phase 2 — All six K cuts + XGBoost-only horizon leaderboard (4-5 days).** Generate matrices for all six K values. Train one XGBoost per K with the existing hyperparameters. Publish the headline accuracy-vs-horizon chart. This is the milestone that justifies the next outreach push.

**Phase 3 — ERA5 swap-in + retrain (3-4 days).** Once ERA5 download completes (background, in parallel with Phase 1-2), swap proxy weather for real ERA5 in the weather/water-balance feature builders. Retrain all six XGBoosts. Expect 1-3 bu/ac RMSE improvement per the README's planned-improvement note.

**Phase 4 — PINN in-season + conformal intervals (3-4 days).** Train PINN variants at K=24, 28, 32, 36. The physics penalty is unchanged. Replace residual-bootstrap intervals with split conformal prediction (calibrated on 2022). Final deliverable: a per-K calibrated forecast pair (XGBoost + PINN) with 90% intervals.

**Total:** ~14-17 working days. Lines up with the strategy doc's "weeks 1-3" estimate.

## Design decisions where I'm guessing — flag before I build

These are the choices I'd make if you don't push back. None of them are obvious wins; happy to defend or change.

1. **Week-K granularity is 4 weeks (16, 20, 24, 28, 32, 36), not 1 week.** Weekly resolution would be 21 model variants between K=16 and K=36 — too many to train, validate, and reason about. Four-week steps land on agronomically meaningful phases (planting, vegetative, pre-silking, silking, grain fill, near-harvest). If you want finer resolution, K=18, 22, 26, 30, 34 could be added later by interpolation rather than retraining.
2. **Train per-K models, not one model conditioned on K.** A single model with K-as-feature is more parameter-efficient but harder to debug for leakage and harder to explain to a buyer ("here is our week-28 model"). Per-K models are simpler and let us version them independently.
3. **Keep the full-season models (module 2 / module 3) intact.** They are the historical leaderboard, the README hindcast result, and the regression test if an in-season model misbehaves. We do not replace them; we add the K models alongside.
4. **Phase 1 ships against the proxy weather, not ERA5.** Real ERA5 is gated on the download finishing. Phase 1 ships earlier value (functioning in-season skeleton against existing weather) and Phase 3 swaps in ERA5 once it's ready. This parallelizes the engineering and the download.
5. **No new model architecture in this build.** XGBoost + PINN, same hyperparameters as full-season. The leverage here is feature engineering and leakage discipline, not model capacity. Once horizon-accuracy curve is established, we can experiment with per-K hyperparameter tuning.

## Out of scope for this build

- Sub-county / field-level forecasting (uploaded shapefiles). Phase 2 of the strategy doc, post-MVP.
- Soybeans. Same data infrastructure; defer.
- Sentinel-2. MODIS at 250m is sufficient for county-level; Sentinel jump is post-seed.
- Hyperparameter search per K. Use full-season hyperparameters initially.
- New states beyond IA/IL/IN. Strategy doc explicitly defers.

## What I need from you before I start

Three things, in order:

1. **Sign-off on the design decisions above (1-5).** If any are wrong, change them before I write code.
2. **Confirm the file layout** — happy with the `cropsight/features/` Python package, or do you prefer to keep everything in notebooks?
3. **Confirm priority of ERA5 swap-in.** Phase 3 is gated on ERA5 download completing. If the download stalls or hits CDS quota issues, do we delay Phase 2 to wait, or ship the leaderboard against proxy weather and update it later? My recommendation: ship against proxy, update later.

Once those three are settled, I start on Phase 1 immediately.
