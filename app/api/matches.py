from __future__ import annotations

import logging

from fastapi import APIRouter, Cookie, Depends, HTTPException, Query, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.players import COOKIE_MAX_AGE, PLAYER_COOKIE
from app.db import get_session
from app.engine import EngineUnavailable
from app.matches import service
from app.matches.service import (
    GameAlreadyOver,
    IllegalMove,
    MatchNotFound,
    NotYourTurn,
)
from app.models.match import Move, Player
from app.schemas.match import (
    MatchCreate,
    MatchRead,
    MoveList,
    MoveRead,
    MoveResponse,
    MoveSubmit,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/matches", tags=["matches"])


def _ensure_player(session: Session, player_id: str | None, response: Response) -> Player:
    if player_id:
        existing = session.get(Player, player_id)
        if existing is not None:
            return existing
    # Auto-create anon player on first match creation so clients don't need
    # a separate onboarding step.
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


@router.post("", response_model=MatchRead, status_code=status.HTTP_201_CREATED)
async def create_match(
    payload: MatchCreate,
    response: Response,
    player_id: str | None = Cookie(default=None, alias=PLAYER_COOKIE),
    session: Session = Depends(get_session),
) -> MatchRead:
    player = _ensure_player(session, player_id, response)

    try:
        match = service.create_match(
            session,
            character_id=payload.character_id,
            player_id=player.id,
            player_color=payload.player_color,
        )
        session.commit()
    except service.MatchError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # If the character is white, play the opening move before returning.
    try:
        await service.start_match_play(session, match)
        session.commit()
    except EngineUnavailable as exc:
        session.rollback()
        raise HTTPException(status_code=503, detail=f"Engine unavailable: {exc}") from exc

    session.refresh(match)
    return MatchRead.model_validate(match)


@router.get("/{match_id}", response_model=MatchRead)
def get_match(match_id: str, session: Session = Depends(get_session)) -> MatchRead:
    try:
        match = service.get_match(session, match_id)
    except MatchNotFound:
        raise HTTPException(status_code=404, detail="Match not found")
    return MatchRead.model_validate(match)


@router.post("/{match_id}/move", response_model=MoveResponse)
async def submit_move(
    match_id: str,
    payload: MoveSubmit,
    session: Session = Depends(get_session),
) -> MoveResponse:
    try:
        player_move, agent_move = await service.apply_player_move(
            session, match_id=match_id, uci=payload.uci
        )
        session.commit()
    except MatchNotFound:
        raise HTTPException(status_code=404, detail="Match not found")
    except IllegalMove as exc:
        session.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except NotYourTurn as exc:
        session.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except GameAlreadyOver as exc:
        session.rollback()
        raise HTTPException(status_code=409, detail=f"Game already over: {exc}") from exc
    except EngineUnavailable as exc:
        session.rollback()
        raise HTTPException(status_code=503, detail=f"Engine unavailable: {exc}") from exc

    match = service.get_match(session, match_id)
    session.refresh(match)

    return MoveResponse(
        match=MatchRead.model_validate(match),
        player_move=MoveRead.model_validate(player_move),
        agent_move=MoveRead.model_validate(agent_move) if agent_move else None,
        game_over=match.status.value != "in_progress",
        outcome=service.player_outcome(match),
    )


@router.post("/{match_id}/resign", response_model=MatchRead)
def resign_match(match_id: str, session: Session = Depends(get_session)) -> MatchRead:
    try:
        match = service.resign(session, match_id=match_id)
        session.commit()
    except MatchNotFound:
        raise HTTPException(status_code=404, detail="Match not found")
    except GameAlreadyOver as exc:
        raise HTTPException(status_code=409, detail=f"Game already over: {exc}") from exc
    return MatchRead.model_validate(match)


@router.get("/{match_id}/moves", response_model=MoveList)
def list_moves(
    match_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(200, ge=1, le=500),
    session: Session = Depends(get_session),
) -> MoveList:
    # Ensure match exists first
    try:
        service.get_match(session, match_id)
    except MatchNotFound:
        raise HTTPException(status_code=404, detail="Match not found")

    rows = list(
        session.execute(
            select(Move)
            .where(Move.match_id == match_id)
            .order_by(Move.move_number)
            .offset(offset)
            .limit(limit)
        ).scalars()
    )
    total = session.query(Move).filter(Move.match_id == match_id).count()
    return MoveList(total=total, items=[MoveRead.model_validate(m) for m in rows])
