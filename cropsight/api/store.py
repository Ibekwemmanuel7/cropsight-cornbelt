"""
Data store for the API. Loads horizon_leaderboard.parquet and the per-K
predictions_k{K}.parquet files at startup; serves point + interval lookups
in O(1) from in-memory dicts.

Single source of truth: data/interim/. The store reads only files produced
by scripts/train_in_season_models.py.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "interim"


class ForecastStore:
    """In-memory predictions indexed by (fips, year, week_k)."""

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        self.leaderboard: pd.DataFrame = pd.DataFrame()
        # predictions[week_k] = DataFrame with columns
        #   fips, year, yield_bu_acre, pred, lower, upper, q_alpha, within_interval
        self.predictions: dict[int, pd.DataFrame] = {}
        self.backend: str = "unknown"
        self.alpha: float = 0.1
        self.load()

    def load(self) -> None:
        leader_path = self.data_dir / "horizon_leaderboard.parquet"
        if not leader_path.exists():
            self.leaderboard = pd.DataFrame()
            self.predictions = {}
            self.backend = "unknown"
            self.alpha = 0.1
            return

        self.leaderboard = pd.read_parquet(leader_path)
        if not self.leaderboard.empty:
            self.backend = str(self.leaderboard["backend"].iloc[0])
            self.alpha = (
                float(self.leaderboard["alpha"].iloc[0])
                if "alpha" in self.leaderboard.columns
                else 0.1
            )

        self.predictions = {}
        for K in sorted(self.leaderboard["week_k"].unique().tolist()):
            path = self.data_dir / f"predictions_k{int(K)}.parquet"
            if path.exists():
                df = pd.read_parquet(path)
                df["fips"] = df["fips"].astype(str)
                df["year"] = df["year"].astype(int)
                self.predictions[int(K)] = df

    @property
    def weeks_available(self) -> list[int]:
        return sorted(self.predictions.keys())

    @property
    def counties(self) -> list[str]:
        """Union of FIPS codes across all K parquets."""
        codes: set[str] = set()
        for df in self.predictions.values():
            codes.update(df["fips"].unique().tolist())
        return sorted(codes)

    def get_forecast(self, fips: str, week_k: int) -> dict | None:
        if week_k not in self.predictions:
            return None
        df = self.predictions[week_k]
        sub = df[df["fips"] == fips]
        if sub.empty:
            return None
        row = sub.iloc[0]
        observed = row.get("yield_bu_acre", np.nan)
        return {
            "fips": str(row["fips"]),
            "year": int(row["year"]),
            "week_k": int(week_k),
            "cutoff_doy": int(week_k) * 7,
            "pred": float(row["pred"]),
            "lower": float(row["lower"]),
            "upper": float(row["upper"]),
            "q_alpha": float(row["q_alpha"]),
            "alpha": self.alpha,
            "backend": self.backend,
            "observed_yield": (
                float(observed) if observed is not None and not pd.isna(observed) else None
            ),
        }

    def get_season(self, fips: str) -> list[dict]:
        out = []
        for K in self.weeks_available:
            f = self.get_forecast(fips, K)
            if f is not None:
                out.append(f)
        return out

    def get_leaderboard(self) -> list[dict]:
        if self.leaderboard.empty:
            return []
        rows = self.leaderboard.copy()
        # NaN coverage on val rows needs to be None for JSON
        for col in ("coverage", "mean_width"):
            if col in rows.columns:
                rows[col] = rows[col].astype(object).where(rows[col].notna(), None)
        return rows.to_dict("records")


def get_store_for_path(path: str | Path | None = None) -> ForecastStore:
    """
    Factory used by tests to pass an alternative data dir. The default
    reads from `cropsight/data/interim/` via REPO_ROOT.
    """
    if path is None:
        env_path = os.environ.get("CROPSIGHT_DATA_DIR")
        path = env_path if env_path else None
    return ForecastStore(Path(path) if path else None)
