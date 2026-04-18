from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Cookie, Depends, Form, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.players import COOKIE_MAX_AGE, PLAYER_COOKIE
from app.characters.openings import OPENINGS
from app.characters.style import style_to_prompt_fragments
from app.db import get_session
from app.engine import EngineUnavailable, available_engines
from app.matches import service as match_service
from app.memory.crud import counts_by_scope, counts_by_type, list_for_character
from app.models.character import Character, CharacterState
from app.models.match import Match, Player
from app.schemas.character import CharacterCreate
from app.schemas.match import MoveRead

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))

router = APIRouter(tags=["web"])


def _run_generation_bg(character_id: str) -> None:
    import logging

    from app.characters.memory_generator import generate_and_store

    try:
        generate_and_store(character_id)
    except Exception:
        logging.getLogger(__name__).exception(
            "Memory generation failed for %s", character_id
        )


@router.get("/", response_class=HTMLResponse)
def index(request: Request, session: Session = Depends(get_session)) -> HTMLResponse:
    chars = list(
        session.execute(
            select(Character)
            .where(Character.deleted_at.is_(None))
            .order_by(Character.is_preset.desc(), Character.created_at.desc())
        ).scalars()
    )
    return templates.TemplateResponse(
        request, "index.html", {"characters": chars}
    )


@router.get("/characters/new", response_class=HTMLResponse)
def new_character_form(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "new.html",
        {"openings": OPENINGS},
    )


@router.post("/characters/new")
def create_character_html(
    request: Request,
    background: BackgroundTasks,
    name: str = Form(...),
    short_description: str = Form(""),
    backstory: str = Form(""),
    avatar_emoji: str = Form("♟️"),
    aggression: int = Form(5),
    risk_tolerance: int = Form(5),
    patience: int = Form(5),
    trash_talk: int = Form(5),
    target_elo: int = Form(1400),
    adaptive: str = Form(""),
    opening_preferences: list[str] = Form(default=[]),
    voice_descriptor: str = Form(""),
    quirks: str = Form(""),
    session: Session = Depends(get_session),
) -> RedirectResponse:
    payload = CharacterCreate(
        name=name,
        short_description=short_description,
        backstory=backstory,
        avatar_emoji=avatar_emoji or "♟️",
        aggression=aggression,
        risk_tolerance=risk_tolerance,
        patience=patience,
        trash_talk=trash_talk,
        target_elo=target_elo,
        adaptive=bool(adaptive),
        opening_preferences=opening_preferences,
        voice_descriptor=voice_descriptor,
        quirks=quirks,
    )
    character = Character(
        name=payload.name,
        short_description=payload.short_description,
        backstory=payload.backstory,
        avatar_emoji=payload.avatar_emoji,
        aggression=payload.aggression,
        risk_tolerance=payload.risk_tolerance,
        patience=payload.patience,
        trash_talk=payload.trash_talk,
        target_elo=payload.target_elo,
        current_elo=payload.target_elo,
        floor_elo=payload.target_elo,
        max_elo=payload.max_elo if payload.max_elo is not None else payload.target_elo + 400,
        adaptive=payload.adaptive,
        opening_preferences=list(payload.opening_preferences),
        voice_descriptor=payload.voice_descriptor,
        quirks=payload.quirks,
        state=CharacterState.GENERATING_MEMORIES,
        memory_generation_started_at=datetime.utcnow(),
        is_preset=False,
    )
    session.add(character)
    session.commit()
    session.refresh(character)
    background.add_task(_run_generation_bg, character.id)

    return RedirectResponse(url=f"/characters/{character.id}", status_code=303)


@router.get("/characters/{character_id}", response_class=HTMLResponse)
def character_detail(
    request: Request,
    character_id: str,
    session: Session = Depends(get_session),
) -> HTMLResponse:
    character = session.get(Character, character_id)
    if character is None or character.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Character not found")

    scope_counts = counts_by_scope(session, character_id=character_id)
    type_counts = counts_by_type(session, character_id=character_id)
    fragments = style_to_prompt_fragments(character)

    # Sample a handful of memories grouped by type for a quick tour.
    samples_by_type: dict[str, list] = {}
    for type_value in type_counts:
        from app.models.memory import MemoryType as MT

        try:
            mt = MT(type_value)
        except ValueError:
            continue
        rows, _ = list_for_character(
            session, character_id=character_id, type_=mt, offset=0, limit=3
        )
        samples_by_type[type_value] = rows

    return templates.TemplateResponse(
        request,
        "detail.html",
        {
            "character": character,
            "scope_counts": scope_counts,
            "type_counts": type_counts,
            "total_memories": sum(scope_counts.values()),
            "fragments": fragments,
            "samples_by_type": samples_by_type,
            "is_generating": character.state == CharacterState.GENERATING_MEMORIES,
            "is_failed": character.state == CharacterState.GENERATION_FAILED,
        },
    )


# ------------------------------ Matches ------------------------------


def _ensure_player_cookie(session: Session, player_id: str | None, response: Response) -> Player:
    if player_id:
        existing = session.get(Player, player_id)
        if existing is not None:
            return existing
    player = Player(display_name="Guest")
    session.add(player)
    session.commit()
    session.refresh(player)
    response.set_cookie(
        key=PLAYER_COOKIE,
        value=player.id,
        max_age=COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax",
    )
    return player


@router.post("/play/{character_id}")
async def start_match_html(
    character_id: str,
    response: Response,
    player_id: str | None = Cookie(default=None, alias=PLAYER_COOKIE),
    session: Session = Depends(get_session),
) -> RedirectResponse:
    character = session.get(Character, character_id)
    if character is None or character.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Character not found")

    redirect = RedirectResponse(url="/", status_code=303)
    player = _ensure_player_cookie(session, player_id, redirect)

    try:
        match = match_service.create_match(
            session,
            character_id=character.id,
            player_id=player.id,
            player_color="random",
        )
        session.commit()
        await match_service.start_match_play(session, match)
        session.commit()
    except EngineUnavailable as exc:
        session.rollback()
        logger.exception("Engine unavailable starting match for %s", character_id)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except match_service.MatchError as exc:
        session.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    redirect.headers["location"] = f"/matches/{match.id}"
    return redirect


@router.get("/matches/{match_id}", response_class=HTMLResponse)
def match_page(
    request: Request,
    match_id: str,
    session: Session = Depends(get_session),
) -> HTMLResponse:
    match = session.get(Match, match_id)
    if match is None:
        raise HTTPException(status_code=404, detail="Match not found")
    character = session.get(Character, match.character_id)

    moves = [
        MoveRead.model_validate(m).model_dump(mode="json") for m in sorted(match.moves, key=lambda m: m.move_number)
    ]

    engines = available_engines()
    real_engines = [e for e in engines if e != "mock"]

    return templates.TemplateResponse(
        request,
        "play.html",
        {
            "match": match,
            "character": character,
            "moves_json": moves,
            "engines_available": engines,
            "has_real_engine": bool(real_engines),
        },
    )
