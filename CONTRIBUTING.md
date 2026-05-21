# Contributing to CropSight CornBelt

Thanks for considering a contribution. This document captures the
expectations and the day-to-day workflow.

## Quick start

```bash
git clone https://github.com/Ibekwemmanuel7/cropsight-cornbelt.git
cd cropsight-cornbelt
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
python -m pytest                    # 34 tests, ~3s on a modern laptop
```

If you plan to touch model training or the feature pipeline you will also
want the `modeling` and `geo` extras (`pip install -e ".[modeling,geo]"`)
plus your own NASS / GEE / CDS credentials. See the
[Installation and setup](README.md#installation-and-setup) section.

## Workflow

### Branches

- `main` is protected and reflects releaseable state. Every push runs CI.
- Feature work happens on short-lived topic branches off `main`. Use
  prefixes that match the change shape: `feat/`, `fix/`, `chore/`, `docs/`,
  `refactor/`, `test/`. Example: `feat/era5-weather-features`.

### Commits

- Imperative subject (`Add water-balance to_week`, not `Added`).
- Reference the spec or issue when applicable
  (`docs/specs/in_season_pipeline.md §2C`, `#42`).
- Squash noise commits before pushing if you can; if not, the PR squash-merge
  will handle it.

### Pull requests

- Open against `main`. Reference the issue or spec section in the body.
- CI must be green before merge. The workflow runs ruff lint, ruff format
  check, and pytest on Python 3.10 / 3.11 / 3.12 — see
  `.github/workflows/ci.yml`.
- For changes that affect modeling accuracy: include a before/after of the
  horizon leaderboard or RMSE numbers in the PR description.
- For changes that touch the leakage discipline: include a new pytest case
  in `tests/test_leakage.py` that would have caught the original bug.

## Code style

- Format and lint with ruff: `python -m ruff format . && python -m ruff check .`
- Pre-commit hooks fire on `git commit` and auto-fix what they can. Re-stage
  the changes and commit again if the hook makes edits.
- Type hints are encouraged but not enforced strictly. Run
  `python -m mypy cropsight` to check if you've changed signatures in the
  public API.
- Keep public functions documented; module docstrings should explain the
  invariant ("what the rule is") not just the mechanics ("what the code
  does").

## Testing

- New feature → new test. Modify an existing feature → update an existing
  test (or add a new one).
- Anything that touches the in-season pipeline must include a leakage check
  if the feature could in principle look at future data.
- Tests under `tests/test_api.py` use the FastAPI `TestClient` and skip if
  `data/interim/horizon_leaderboard.parquet` is absent. Build the parquet
  locally with `scripts/build_in_season_features.py` and
  `scripts/train_in_season_models.py` before running the full suite.

## What goes where

| Concern | Location |
|---|---|
| Feature engineering | `cropsight/features/` |
| Uncertainty / calibration | `cropsight/uncertainty/` |
| Public API | `cropsight/api/` |
| One-shot pipeline drivers | `scripts/` |
| Notebook research | repo root `module*.ipynb` |
| Specs and design docs | `docs/specs/` |
| Tests | `tests/` |

## Reporting bugs and requesting features

Open an issue using the templates under `.github/ISSUE_TEMPLATE/`. The bug
template asks for a minimal repro, the expected vs observed behaviour, and
the environment (OS / Python / package extras). The feature template asks
for the motivating use case so the maintainers can scope it.

## Communication

For now, GitHub Issues and PRs are the only channel. Critical security
reports should go to the email in the repo About box (please don't open a
public issue for security findings).

## License

By contributing you agree your contributions will be licensed under the MIT
License (see [LICENSE](LICENSE)).
