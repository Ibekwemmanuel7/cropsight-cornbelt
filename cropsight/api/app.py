"""
FastAPI app exposing in-season forecast lookups.

Endpoints
---------
GET  /health
GET  /counties
GET  /leaderboard
GET  /forecast?fips=<fips>&week=<K>
GET  /forecast/county/{fips}/season

Auto-generated OpenAPI docs at /docs.

Run with:
    uvicorn cropsight.api.app:app --reload
or
    python scripts/serve_api.py
"""

from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query

from . import schemas
from .store import ForecastStore, get_store_for_path

_store: ForecastStore | None = None


def get_store() -> ForecastStore:
    global _store
    if _store is None:
        _store = get_store_for_path()
    return _store


def create_app(data_dir: str | Path | None = None) -> FastAPI:
    """
    Build a FastAPI app. Optionally point at a custom data dir (used in tests).
    """
    global _store
    _store = get_store_for_path(data_dir)

    app = FastAPI(
        title="CropSight CornBelt — In-Season Forecast API",
        version="0.1.0",
        description=(
            "Read-only API exposing county-level in-season corn yield "
            "forecasts for IA / IL / IN with split-conformal 90% intervals. "
            "Phase 1 + Phase 2 + Phase 4-prep deliverable; the underlying "
            "model is phenology-only (Phase 3 ERA5 swap-in pending)."
        ),
    )

    @app.get("/health", response_model=schemas.Health, tags=["meta"])
    def health(store: ForecastStore = Depends(get_store)):
        n_counties = len(store.counties)
        return schemas.Health(
            status="ok" if n_counties > 0 else "no_data",
            n_counties=n_counties,
            weeks_available=store.weeks_available,
            backend=store.backend,
            note=(
                "Preview leaderboard - rerun train_in_season_models.py with "
                "xgboost installed for production numbers."
                if store.backend != "xgboost"
                else None
            ),
        )

    @app.get("/counties", response_model=schemas.CountyList, tags=["meta"])
    def counties(store: ForecastStore = Depends(get_store)):
        codes = store.counties
        return schemas.CountyList(fips_codes=codes, n=len(codes))

    @app.get("/leaderboard", response_model=schemas.Leaderboard, tags=["meta"])
    def leaderboard(store: ForecastStore = Depends(get_store)):
        rows = store.get_leaderboard()
        if not rows:
            raise HTTPException(status_code=503, detail="No leaderboard data loaded.")
        return schemas.Leaderboard(
            rows=[schemas.LeaderboardRow(**r) for r in rows],
            target_coverage=1.0 - store.alpha,
        )

    @app.get("/forecast", response_model=schemas.Forecast, tags=["forecast"])
    def forecast(
        fips: str = Query(..., min_length=5, max_length=5, description="5-digit county FIPS code."),
        week: int = Query(..., description="Forecast week K (16..40)."),
        store: ForecastStore = Depends(get_store),
    ):
        if week not in store.weeks_available:
            raise HTTPException(
                status_code=400,
                detail=f"week K={week} not available. Available: {store.weeks_available}",
            )
        result = store.get_forecast(fips, week)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"no forecast for fips={fips} at week K={week}",
            )
        return schemas.Forecast(**result)

    @app.get(
        "/forecast/county/{fips}/season",
        response_model=schemas.SeasonForecast,
        tags=["forecast"],
    )
    def forecast_county_season(
        fips: str,
        store: ForecastStore = Depends(get_store),
    ):
        if len(fips) != 5:
            raise HTTPException(status_code=400, detail="fips must be a 5-digit county code.")
        forecasts_raw = store.get_season(fips)
        if not forecasts_raw:
            raise HTTPException(status_code=404, detail=f"no forecasts for fips={fips}")
        year = forecasts_raw[0]["year"]
        return schemas.SeasonForecast(
            fips=fips,
            year=year,
            forecasts=[schemas.Forecast(**f) for f in forecasts_raw],
        )

    return app


# Module-level app for `uvicorn cropsight.api.app:app`
app = create_app()
