# Changelog

All notable changes to CropSight CornBelt are documented in this file. Format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned (Phase 3 — blocked on ERA5 download)
- Implement `cropsight.features.weather` against real ERA5 daily T2m, Td2m,
  total precip, and SSRD aggregated to county centroids.
- Implement `cropsight.features.water_balance.run_water_balance_to_week` with
  real PET inputs from Hargreaves-Samani on ERA5 SSRD.
- Re-train all six K models and refresh the horizon leaderboard.

### Planned (Phase 4 — PINN in-season)
- Per-K PINN variants with the unchanged water-stress physics penalty.
- A combined leaderboard comparing XGBoost-at-K to PINN-at-K.

### Planned (product surface)
- Next.js dashboard with county choropleth and per-county forecast trajectory.
- Auth (Clerk/Auth.js) + Stripe billing for the agronomist tier.
- Client-report PDF generator.

---

## [0.1.0] — 2026-05-21

First tagged release. End-to-end in-season corn yield forecasting for IA / IL /
IN counties with split-conformal 90% prediction intervals and a read-only
FastAPI service. The accuracy curve is bounded by proxy weather features; real
ERA5 swap-in is in flight (Phase 3).

### Added
- **`cropsight/features/`** — package for in-season feature engineering.
  - `phenology.py`: `smooth_truncated`, `extract_phenology_to_week`,
    `compute_vci_to_week`, `vci_window_means_to_week`.
  - `leakage.py`: `LeakageError`, `audit_no_future_data`,
    `audit_static_features`, `audit_train_val_test_split`,
    `audit_ndvi_input_doy`, `audit_vci_baseline`.
  - `weather.py` and `water_balance.py` — Phase 3 stubs that raise
    `WeatherPhase3Stub` / `WaterBalancePhase3Stub`.
- **`cropsight/uncertainty/conformal.py`** — split conformal prediction with
  the standard `ceil((n+1)(1-α))` quantile rank. Year-based calibration split
  (2000–2019 train, 2020–2021 calibrate, 2022 val, 2023 test).
- **`cropsight/api/`** — FastAPI service. Endpoints: `/health`, `/counties`,
  `/leaderboard`, `/forecast`, `/forecast/county/{fips}/season`. Auto OpenAPI
  docs at `/docs`. In-memory `ForecastStore` loads parquets on startup.
- **Scripts** — `build_in_season_features.py` (six K matrices),
  `train_in_season_models.py` (per-K XGBoost + conformal), `serve_api.py`
  (uvicorn runner), `plot_horizon_leaderboard.py`, `download_era5.py`.
- **CDL pixel masking** wired into module 1's NDVI extraction. Was previously
  documented in the README but not actually applied. Pre-2008 fallback uses a
  "stable corn" mask (pixels classified as corn ≥3 years in 2008–2023).
- **Project hygiene** — `pyproject.toml` with optional extras (`modeling`,
  `geo`, `viz`, `dev`, `all`), GitHub Actions CI on Python 3.10/3.11/3.12,
  ruff lint + format, pre-commit hooks, mypy config.
- **Docker** — multi-stage Dockerfile, `.dockerignore`, `docker-compose.yml`
  for local stack with read-only data mount.
- **Docs** — `docs/specs/in_season_pipeline.md` (3-page tech spec),
  `cropsight_strategy.docx` (investor-grade strategy doc), embedded preview
  PNGs in the README, four new images in `docs/images/`.
- **Tests** — 34 pytest cases across phenology, leakage, conformal, and API
  modules; CI runs lint + format + tests on every push and PR.

### Changed
- Stacking ensemble dropped from the headline pipeline. Empirical null result
  (test RMSE 14.0, worse than every base model). The README now leads with
  XGBoost + PINN as the calibrated production pair; the ensemble code is
  retained in `module3_modeling.ipynb` as a documented negative finding.
- README results table reorganized to lead with the production pair and to
  include the in-season horizon leaderboard.
- Notebook paths fixed (root-level, not `notebooks/`) and `YOUR_USERNAME`
  placeholders replaced with the actual GitHub handle in the clone command
  and citation block.
- Residual-bootstrap intervals replaced by split conformal across all six K
  cuts. Coverage is now calibrated and MRM-reviewable.

### Fixed
- Broken LICENSE / `requirements.txt` references in the README (the files
  themselves did not exist in the repo until v0.1.0).
- Numerous Ruff lint findings across the codebase; the project now passes
  `ruff check .` and `ruff format --check .` clean.

### Honest caveats
- The Phase 2 leaderboard uses sklearn `HistGradientBoostingRegressor` as a
  stand-in while xgboost is unavailable in the sandbox. Re-run with xgboost
  installed for production numbers.
- Weather and water-balance features are NOT yet in the in-season matrices;
  Phase 3 ERA5 swap-in pending the download.
- The full-season hindcast (12.5 RMSE, XGBoost) outperforms the in-season K
  models on absolute accuracy because the in-season training set holds out
  2020–2021 for conformal calibration. This is the correct tradeoff for
  enterprise / insurance pitches that need coverage guarantees.

[Unreleased]: https://github.com/Ibekwemmanuel7/cropsight-cornbelt/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Ibekwemmanuel7/cropsight-cornbelt/releases/tag/v0.1.0
