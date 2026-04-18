from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.characters import router as characters_api
from app.api.matches import router as matches_api
from app.api.players import router as players_api
from app.characters.seed import seed_presets
from app.config import get_settings
from app.db import init_db
from app.logging_config import configure_logging
from app.web.routes import router as web_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    init_db()

    settings = get_settings()
    # Only kick off preset memory generation if the API key is present — the app
    # should still come up locally without one so people can inspect the UI.
    run_gen = bool(settings.gemini_api_key)
    results = seed_presets(run_generation=run_gen)
    created = [k for k, v in results.items() if v]
    if created:
        logger.info("Seeded new presets: %s", created)
    if not run_gen:
        logger.warning(
            "GEMINI_API_KEY is not set — presets were seeded but memory generation is skipped."
        )
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Metropolis Chess Club", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory="app/web/static"), name="static")
    app.include_router(characters_api)
    app.include_router(players_api)
    app.include_router(matches_api)
    app.include_router(web_router)
    return app


app = create_app()
