"""
Tests for cropsight.api.

These use FastAPI's TestClient and the real data files in data/interim/
produced by scripts/train_in_season_models.py. If those files are absent
the suite is skipped with a clear message.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cropsight.api import create_app
from cropsight.api.store import DEFAULT_DATA_DIR

_LEADERBOARD_EXISTS = (DEFAULT_DATA_DIR / "horizon_leaderboard.parquet").exists()

pytestmark = pytest.mark.skipif(
    not _LEADERBOARD_EXISTS,
    reason="leaderboard not built - run scripts/train_in_season_models.py first",
)


@pytest.fixture(scope="module")
def client():
    app = create_app()
    return TestClient(app)


def test_health_returns_ok_with_counties(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["n_counties"] > 0
    assert sorted(body["weeks_available"]) == [16, 20, 24, 28, 32, 36]
    assert body["backend"] in ("xgboost", "sklearn_hgb")


def test_counties_returns_list(client):
    r = client.get("/counties")
    assert r.status_code == 200
    body = r.json()
    assert body["n"] > 0
    assert body["n"] == len(body["fips_codes"])
    assert all(len(f) == 5 for f in body["fips_codes"])


def test_leaderboard_returns_rows(client):
    r = client.get("/leaderboard")
    assert r.status_code == 200
    body = r.json()
    assert "rows" in body
    assert len(body["rows"]) == 12  # 6 weeks * (val + test)
    assert body["target_coverage"] == pytest.approx(0.9, abs=1e-6)
    test_rows = [r for r in body["rows"] if r["split"] == "test"]
    assert all(r["coverage"] is not None for r in test_rows)


def test_forecast_happy_path(client):
    # Pick a known IA county - Story County, FIPS 19169
    counties_resp = client.get("/counties").json()
    fips = next((c for c in counties_resp["fips_codes"] if c.startswith("19")), None)
    assert fips is not None, "no Iowa counties found in store"

    r = client.get(f"/forecast?fips={fips}&week=28")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["fips"] == fips
    assert body["week_k"] == 28
    assert body["cutoff_doy"] == 196
    assert body["lower"] <= body["pred"] <= body["upper"]
    assert body["q_alpha"] > 0


def test_forecast_unknown_fips_returns_404(client):
    r = client.get("/forecast?fips=99999&week=28")
    assert r.status_code == 404


def test_forecast_invalid_week_returns_400(client):
    r = client.get("/forecast?fips=19169&week=99")
    assert r.status_code == 400


def test_forecast_missing_fips_returns_422(client):
    """FastAPI validation kicks in on missing required query param."""
    r = client.get("/forecast?week=28")
    assert r.status_code == 422


def test_forecast_county_season(client):
    counties_resp = client.get("/counties").json()
    fips = next((c for c in counties_resp["fips_codes"] if c.startswith("19")), None)
    r = client.get(f"/forecast/county/{fips}/season")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["fips"] == fips
    weeks = [f["week_k"] for f in body["forecasts"]]
    assert sorted(weeks) == [16, 20, 24, 28, 32, 36]


def test_forecast_county_season_unknown_fips(client):
    r = client.get("/forecast/county/99999/season")
    assert r.status_code == 404


def test_openapi_schema(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    schema = r.json()
    paths = set(schema["paths"].keys())
    assert "/health" in paths
    assert "/forecast" in paths
    assert "/forecast/county/{fips}/season" in paths
