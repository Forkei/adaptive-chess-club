"""Player-related API endpoints.

Phase 3a replaces the anonymous-cookie model with username-based login.
- `POST /api/players` is kept for back-compat with older clients: it now
  auto-generates a guest username and sets the cookie.
- `GET /api/me` returns the current player (401 if no cookie) — the
  canonical endpoint going forward.
- `GET /api/players/me` stays as an alias for back-compat but behaves
  identically.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response
from sqlalchemy.orm import Session

from app.auth import (
    COOKIE_MAX_AGE,
    PLAYER_COOKIE,
    generate_guest_username,
    require_player,
)
from app.db import get_session
from app.models.match import Player
from app.schemas.match import PlayerCreate, PlayerRead

router = APIRouter(tags=["players"])


@router.post("/api/players", response_model=PlayerRead)
def create_player(
    payload: PlayerCreate,
    response: Response,
    session: Session = Depends(get_session),
) -> PlayerRead:
    """Create a guest player. Phase 3a: assigns an auto-generated
    `guest_<short>` username so the row satisfies the new UNIQUE
    constraint. Real usernames are claimed via POST /login.
    """
    username = generate_guest_username()
    player = Player(username=username, display_name=payload.display_name)
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
    return PlayerRead.model_validate(player)


@router.get("/api/me", response_model=PlayerRead)
def get_me(player: Player = Depends(require_player)) -> PlayerRead:
    return PlayerRead.model_validate(player)


@router.get("/api/players/me", response_model=PlayerRead)
def get_players_me(player: Player = Depends(require_player)) -> PlayerRead:
    """Back-compat alias for /api/me."""
    return PlayerRead.model_validate(player)
