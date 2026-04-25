# Metropolis Chess Club — production image (Hugging Face Spaces + local Docker).
# Base: python:3.12-slim. Installs Stockfish from Debian, Python deps via pip,
# and pre-warms Maia-2 + sentence-transformers weights so cold starts don't stall.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps:
# - stockfish: chess engine binary. /usr/games/stockfish on Debian.
# - build-essential + git: torch transitive deps sometimes need compilation.
# - curl: diagnostics.
RUN apt-get update && apt-get install -y --no-install-recommends \
        stockfish \
        build-essential \
        git \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so Docker caches them across code changes.
COPY pyproject.toml /app/
RUN pip install --upgrade pip && \
    pip install -e "." && \
    pip install "torch>=2.2.0" --index-url https://download.pytorch.org/whl/cpu && \
    pip install \
        "maia2>=0.1.0" \
        "numpy>=1.24.0" \
        "tqdm>=4.65.0" \
        "pandas>=2.0.0" \
        "einops>=0.7.0" \
        "huggingface_hub>=0.20.0" \
        "safetensors>=0.4.0" \
        "scikit-learn>=1.3.0" \
        "pyzstd>=0.16.0" \
        "gdown>=5.0.0"

# Copy app + support files.
COPY app /app/app
COPY scripts /app/scripts
COPY alembic /app/alembic
COPY alembic.ini /app/alembic.ini
COPY pytest.ini /app/
COPY tests /app/tests

# Pre-download model weights at build time so cold starts don't hang.
# HF_HOME is the unified cache for both huggingface_hub (Maia-2) and
# sentence-transformers. MAIA2_CACHE_DIR is a separate location for the
# raw weight files that maia2's own loader expects.
# STOCKFISH_PATH points to the Debian-packaged binary.
ENV MAIA2_CACHE_DIR=/app/maia2_models \
    HF_HOME=/app/.cache/hf \
    STOCKFISH_PATH=/usr/games/stockfish

RUN mkdir -p /app/maia2_models /app/.cache/hf && \
    (python scripts/setup_engines.py || echo "[build] Maia-2 pre-warm failed — weights will download on first use") && \
    (python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); print('[build] sentence-transformers pre-warm OK')" \
        || echo "[build] sentence-transformers pre-warm failed — model will download on first use")

# Runtime defaults.
# DATABASE_URL → HF Spaces persistent volume at /data.
# REDIS_URL is intentionally empty: single-worker HF deployment uses the in-process fallback.
# Override DATABASE_URL in docker-compose for local dev (see docker-compose.yml).
ENV DATABASE_URL=sqlite:////data/metropolis_chess.db \
    LOG_DIR=/app/logs \
    REDIS_URL=""

# Create writable directories. /app/logs must be writable by uid 1000 (HF Spaces
# default runtime user). Model cache dirs are read-only at runtime (pre-baked above).
# /data is created here so the path exists if no volume is mounted (local dev);
# on HF Spaces, the persistent volume overlay provides the real writable /data.
RUN mkdir -p /app/logs /app/data /data && \
    chmod -R 777 /app/logs /app/data /data

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 7860

ENTRYPOINT ["/app/entrypoint.sh"]
