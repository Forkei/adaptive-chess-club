from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api.characters import router as characters_api
from app.api.discovery import router as discovery_api
from app.api.matches import router as matches_api
from app.api.players import router as players_api
from app.auth import NotAuthenticated
from app.characters.seed import seed_presets
from app.config import get_settings
from app.db import init_db
from app.logging_config import configure_logging
from app.sockets import build_asgi_app
from app.sockets.bridge import set_main_loop
from app.web.lobby_routes import router as lobby_router
from app.web.routes import router as web_router

logger = logging.getLogger(__name__)


def _warn_if_multiworker() -> None:
    """Log (but do not fail) when multiple workers look likely.

    Phase 3b assumes a single in-process room + cooldown registry. Multi-worker
    requires Redis pub/sub (Phase 4). A dev running `uvicorn --workers 4` out of
    habit shouldn't have the app refuse to boot — just warn.
    """
    raw = os.environ.get("WEB_CONCURRENCY") or os.environ.get("UVICORN_WORKERS")
    try:
        workers = int(raw) if raw else 1
    except ValueError:
        workers = 1
    if workers > 1:
        logger.warning(
            "Socket.IO disconnect tracking and per-match rooms assume single-worker "
            "deployment (detected WEB_CONCURRENCY/UVICORN_WORKERS=%s). Multi-worker "
            "requires Redis pub/sub (Phase 4). Running anyway — match state may be "
            "inconsistent under load.",
            workers,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    init_db()

    # Capture the main event loop so the post-match processor's daemon thread
    # can schedule Socket.IO emits back onto us via run_coroutine_threadsafe.
    set_main_loop(asyncio.get_running_loop())
    _warn_if_multiworker()

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

    # Housekeeping: reap stale matches, fail stuck analyses, re-arm disconnect
    # cooldowns that survived the restart. Runs once here, then on a timer.
    from app.matches import housekeeping

    try:
        await housekeeping.run_startup()
    except Exception:
        logger.exception("Housekeeping startup sweep failed (continuing)")
    housekeeping_task = asyncio.create_task(
        housekeeping.periodic_loop(), name="match-housekeeping",
    )
    pvp_flagfall_task = asyncio.create_task(
        housekeeping.pvp_flagfall_loop(), name="pvp-flagfall",
    )
    yield

    # Stop the periodic sweeps cleanly before tearing the loop down.
    for t in (housekeeping_task, pvp_flagfall_task):
        t.cancel()
    for t in (housekeeping_task, pvp_flagfall_task):
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass
    # Clear on shutdown so background threads don't try to schedule onto a dead loop.
    set_main_loop(None)


def create_app() -> FastAPI:
    app = FastAPI(title="Metropolis Chess Club", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory="app/web/static"), name="static")
    app.include_router(characters_api)
    app.include_router(players_api)
    app.include_router(matches_api)
    app.include_router(discovery_api)
    app.include_router(lobby_router)
    app.include_router(web_router)

    @app.exception_handler(NotAuthenticated)
    async def _not_authenticated(request: Request, exc: NotAuthenticated):
        # API callers expect JSON 401. HTML callers get a redirect to /login.
        if request.url.path.startswith("/api/"):
            return JSONResponse({"detail": "Not authenticated"}, status_code=401)
        # Preserve where they were trying to go.
        next_url = request.url.path
        if request.url.query:
            next_url = f"{next_url}?{request.url.query}"
        return RedirectResponse(url=f"/login?next={next_url}", status_code=303)

    return app


fastapi_app = create_app()


@fastapi_app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


# The combined ASGI app — uvicorn entry point. HTTP flows through FastAPI;
# Socket.IO traffic is handled by `sio` from app.sockets.server.
app = build_asgi_app(fastapi_app)
