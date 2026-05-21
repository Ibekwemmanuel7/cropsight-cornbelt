"""
cropsight.api
=============

FastAPI service exposing the in-season forecast leaderboard and per-county
predictions. Read-only by design; suitable for demos and enterprise
integration tests. Production deployment notes in scripts/serve_api.py.
"""

from .app import create_app

__all__ = ["create_app"]
