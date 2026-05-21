#!/usr/bin/env python3
"""
Convenience runner for the CropSight forecast API. Wraps uvicorn with sensible
defaults for local development.

Usage:
    python scripts/serve_api.py                 # bind to 127.0.0.1:8000
    python scripts/serve_api.py --host 0.0.0.0  # bind all interfaces
    python scripts/serve_api.py --port 8080
    python scripts/serve_api.py --reload        # auto-reload on file change

Once running, open http://localhost:8000/docs for the auto-generated Swagger UI,
or curl examples:
    curl 'http://localhost:8000/health'
    curl 'http://localhost:8000/forecast?fips=19169&week=28'
    curl 'http://localhost:8000/forecast/county/19169/season'
    curl 'http://localhost:8000/leaderboard'

Production note: this runner is for development. For deployment use a
production ASGI server (uvicorn --workers, hypercorn, or behind nginx).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true",
                        help="Auto-reload on source changes (development only).")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        sys.exit("uvicorn is not installed. Run: pip install fastapi uvicorn")

    uvicorn.run(
        "cropsight.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
