"""Pydantic response models for the cropsight API."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class Health(BaseModel):
    status: str
    n_counties: int
    weeks_available: list[int]
    backend: str
    note: Optional[str] = None


class Forecast(BaseModel):
    fips: str = Field(..., description="5-digit county FIPS code.")
    year: int = Field(..., description="Forecast target year.")
    week_k: int = Field(..., description="Forecast week K (DOY cutoff = K * 7).")
    cutoff_doy: int
    pred: float = Field(..., description="Point forecast, bu/acre.")
    lower: float = Field(..., description="Lower bound of (1 - alpha) interval.")
    upper: float = Field(..., description="Upper bound of (1 - alpha) interval.")
    q_alpha: float = Field(..., description="Conformal interval half-width (bu/acre).")
    alpha: float = Field(..., description="Miscoverage rate, e.g. 0.1 for 90% interval.")
    backend: str = Field(..., description="Model backend used (xgboost / sklearn_hgb).")
    observed_yield: Optional[float] = Field(
        None, description="USDA NASS actual yield, if known."
    )


class SeasonForecast(BaseModel):
    fips: str
    year: int
    forecasts: list[Forecast]


class LeaderboardRow(BaseModel):
    week_k: int
    cutoff_doy: int
    split: str
    rmse: float
    mae: float
    r2: float
    coverage: Optional[float] = None
    mean_width: Optional[float] = None
    n_features: int
    backend: str
    alpha: float


class Leaderboard(BaseModel):
    rows: list[LeaderboardRow]
    target_coverage: float


class CountyList(BaseModel):
    fips_codes: list[str]
    n: int
