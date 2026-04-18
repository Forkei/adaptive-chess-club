# Metropolis Chess Club — Phase 2a image.
# Base: python:3.12-slim. Installs Stockfish from Debian, Python deps via pip,
# and pre-warms Maia-2 weights so first inference after boot is fast.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps:
# - stockfish: the engine binary. Debian's package is old-ish but works fine for
#   our Elo range.
# - build-essential + git: some ML deps (torch transitive) sometimes need to
#   compile. Cheap to install and keeps build deterministic.
# - curl: diagnostics.
RUN apt-get update && apt-get install -y --no-install-recommends \
        stockfish \
        build-essential \
        git \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so docker caches them across code changes.
COPY pyproject.toml /app/
RUN pip install --upgrade pip && \
    pip install -e "." && \
    pip install "torch>=2.2.0" --index-url https://download.pytorch.org/whl/cpu && \
    pip install "maia2>=1.0.0"

# Copy the app after deps — this layer churns per commit.
COPY app /app/app
COPY scripts /app/scripts
COPY pytest.ini /app/
COPY tests /app/tests

# Warm the Maia-2 cache. Non-fatal — if weights can't be fetched at build time,
# they'll download on first inference instead.
ENV MAIA2_CACHE_DIR=/app/.cache/maia2 \
    HF_HOME=/app/.cache/maia2 \
    STOCKFISH_PATH=/usr/games/stockfish
RUN mkdir -p /app/.cache/maia2 && \
    (python scripts/setup_engines.py || echo "engine pre-warm failed; continuing")

# App runtime
ENV DATABASE_URL=sqlite:////app/data/metropolis_chess.db \
    LOG_DIR=/app/logs \
    REDIS_URL=redis://redis:6379/0
RUN mkdir -p /app/data /app/logs

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
