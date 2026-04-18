from __future__ import annotations

from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.characters.openings import OPENINGS
from app.characters.style import style_to_prompt_fragments
from app.db import get_session
from app.memory.crud import counts_by_scope, counts_by_type, list_for_character
from app.models.character import Character, CharacterState
from app.schemas.character import CharacterCreate

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
