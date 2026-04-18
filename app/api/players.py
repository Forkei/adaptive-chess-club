"""Cookie-based anonymous players.

No auth. `POST /api/players` returns a Player and sets a `player_id`
cookie. `GET /api/players/me` reads that cookie and returns the row;
if the cookie is missing/unknown, it returns 404 — clients should then
call POST.
"""

from __future__ import annotations

from fastapi import APIRouter, Cookie, Depends, HTTPException, Response
from sqlalchemy.orm import Session

from app.db import get_session
from app.models.match import Player
from app.schemas.match import PlayerCreate, PlayerRead

router = APIRouter(prefix="/api/players", tags=["players"])

PLAYER_COOKIE = "player_id"
COOKIE_MAX_AGE = 60 * 60 * 24 * 365  # one year


@router.post("", response_model=PlayerRead)
def create_player(
    payload: PlayerCreate,
    response: Response,
    session: Session = Depends(get_session),
) -> PlayerRead:
    player = Player(display_name=payload.display_name)
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


@router.get("/me", response_model=PlayerRead)
def get_me(
    player_id: str | None = Cookie(default=None, alias=PLAYER_COOKIE),
    session: Session = Depends(get_session),
) -> PlayerRead:
    if not player_id:
        raise HTTPException(status_code=404, detail="No player cookie set")
    player = session.get(Player, player_id)
    if player is None:
        raise HTTPException(status_code=404, detail="Player not found")
    return PlayerRead.model_validate(player)
