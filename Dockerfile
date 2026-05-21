# CropSight CornBelt - read-only forecast API container
#
# Multi-stage build keeps the runtime image small:
#   - builder stage installs everything into a venv
#   - runtime stage copies the venv only and runs as non-root
#
# Build  : docker build -t cropsight-api .
# Run    : docker run --rm -p 8000:8000 -v "$(pwd)/data:/app/data:ro" cropsight-api
# Probe  : curl http://localhost:8000/health

ARG PYTHON_VERSION=3.12

# ---- builder ---------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# System build deps. Kept minimal because the API tier doesn't need
# torch / Earth Engine / rasterio.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv "$VIRTUAL_ENV"

WORKDIR /build

# Copy only what's needed to resolve dependencies first - keeps layer cache hot
COPY pyproject.toml README.md ./
COPY cropsight ./cropsight

# Install the package + modeling extra (XGBoost is needed at predict time).
# The geo + viz extras are left out of the API image - they're notebook deps.
RUN pip install --upgrade pip \
 && pip install ".[modeling]"

# ---- runtime ---------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    CROPSIGHT_DATA_DIR=/app/data/interim

# libgomp1 needed by xgboost at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash --uid 1001 cropsight

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Copy only the runtime parts. Tests, notebooks, scripts/, docs/ are not
# needed in the image.
COPY --chown=cropsight:cropsight cropsight ./cropsight
COPY --chown=cropsight:cropsight pyproject.toml README.md LICENSE ./

USER cropsight
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl --fail --silent http://localhost:8000/health || exit 1

CMD ["uvicorn", "cropsight.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
