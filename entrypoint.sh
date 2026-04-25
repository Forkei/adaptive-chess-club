#!/usr/bin/env bash
# Boot entrypoint for Metropolis Chess Club.
# 1. Runs schema migration (create_all on fresh DB; alembic upgrade head on existing).
# 2. Starts uvicorn on 0.0.0.0:7860 (HF Spaces app_port).
set -e

python scripts/migrate.py
exec uvicorn app.main:app --host 0.0.0.0 --port 7860
