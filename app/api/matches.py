from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth import require_player
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
from app.post_match.processor import start_post_match_background
from app.schemas.match import (
    AgentTurnInfo,
    GeneratedMemorySnippet,
    MatchCreate,
    MatchRead,
    MoveList,
    MoveRead,
    MoveResponse,
    MoveSubmit,
    PostMatchStatus,
    SurfacedMemorySnippet,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/matches", tags=["matches"])


@router.post("", response_model=MatchRead, status_code=status.HTTP_201_CREATED)
async def create_match(
    payload: MatchCreate,
    player: Player = Depends(require_player),
    session: Session = Depends(get_session),
) -> MatchRead:
    # Phase 3a: enforce visibility + content rating on match start.
    from app.models.character import Character, ContentRating, Visibility, rating_allowed

    character = session.get(Character, payload.character_id)
    if character is None or character.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Character not found")
    if character.visibility == Visibility.PRIVATE and character.owner_id != player.id:
        raise HTTPException(status_code=404, detail="Character not found")
    if not rating_allowed(character.content_rating, player.max_content_rating):
        raise HTTPException(
            status_code=403,
            detail="This character's content rating exceeds your preference.",
        )

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


def _agent_turn_info(outcome) -> AgentTurnInfo | None:
    if outcome is None:
        return None
    snippets = [
        SurfacedMemorySnippet(
            memory_id=m.memory_id,
            narrative_text=m.narrative_text,
            retrieval_reason=m.retrieval_reason,
            from_cache=m.from_cache,
        )
        for m in outcome.surfaced
    ]
    return AgentTurnInfo(
        speak=outcome.soul.speak,
        emotion=outcome.soul.emotion,
        emotion_intensity=outcome.soul.emotion_intensity,
        surfaced_memories=snippets,
    )


@router.get("/{match_id}", response_model=MatchRead)
def get_match(
    match_id: str,
    player: Player = Depends(require_player),
    session: Session = Depends(get_session),
) -> MatchRead:
    try:
        match = service.get_match(session, match_id)
    except MatchNotFound:
        raise HTTPException(status_code=404, detail="Match not found")
    if match.player_id != player.id:
        raise HTTPException(status_code=404, detail="Match not found")
    return MatchRead.model_validate(match)


@router.post("/{match_id}/move", response_model=MoveResponse)
async def submit_move(
    match_id: str,
    payload: MoveSubmit,
    player: Player = Depends(require_player),
    session: Session = Depends(get_session),
) -> MoveResponse:
    # Match must belong to the logged-in player.
    try:
        m = service.get_match(session, match_id)
    except MatchNotFound:
        raise HTTPException(status_code=404, detail="Match not found")
    if m.player_id != player.id:
        raise HTTPException(status_code=404, detail="Match not found")
    try:
        player_move, agent_move, agent_outcome = await service.apply_player_move(
            session, match_id=match_id, uci=payload.uci, player_chat=payload.chat
        )
        session.commit()
        # If the move finalized the match, kick off post-match processing.
        # Safe to spawn AFTER commit so the background thread can read a
        # committed match row.
        match_post_commit = service.get_match(session, match_id)
        if match_post_commit.status.value != "in_progress":
            start_post_match_background(match_id)
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
        agent_turn=_agent_turn_info(agent_outcome),
        game_over=match.status.value != "in_progress",
        outcome=service.player_outcome(match),
    )


@router.post("/{match_id}/resign", response_model=MatchRead)
def resign_match(
    match_id: str,
    player: Player = Depends(require_player),
    session: Session = Depends(get_session),
) -> MatchRead:
    try:
        match_check = service.get_match(session, match_id)
    except MatchNotFound:
        raise HTTPException(status_code=404, detail="Match not found")
    if match_check.player_id != player.id:
        raise HTTPException(status_code=404, detail="Match not found")
    try:
        match = service.resign(session, match_id=match_id)
        session.commit()
    except MatchNotFound:
        raise HTTPException(status_code=404, detail="Match not found")
    except GameAlreadyOver as exc:
        raise HTTPException(status_code=409, detail=f"Game already over: {exc}") from exc
    start_post_match_background(match_id)
    return MatchRead.model_validate(match)


@router.get("/{match_id}/post_match_status", response_model=PostMatchStatus)
def post_match_status(
    match_id: str,
    player: Player = Depends(require_player),
    session: Session = Depends(get_session),
) -> PostMatchStatus:
    from app.models.match import MatchAnalysis
    from app.models.memory import Memory
    from app.models.match import OpponentProfile

    try:
        match = service.get_match(session, match_id)
    except MatchNotFound:
        raise HTTPException(status_code=404, detail="Match not found")
    if match.player_id != player.id:
        raise HTTPException(status_code=404, detail="Match not found")

    analysis = session.execute(
        select(MatchAnalysis).where(MatchAnalysis.match_id == match_id)
    ).scalar_one_or_none()

    if analysis is None:
        return PostMatchStatus(match_id=match_id, status="none")

    # Hydrate generated memories if any.
    memory_snippets: list[GeneratedMemorySnippet] = []
    if analysis.generated_memory_ids:
        rows = list(
            session.execute(
                select(Memory).where(Memory.id.in_(list(analysis.generated_memory_ids)))
            ).scalars()
        )
        memory_snippets = [
            GeneratedMemorySnippet(
                memory_id=r.id,
                narrative_text=r.narrative_text,
                triggers=list(r.triggers or []),
                emotional_valence=float(r.emotional_valence),
            )
            for r in rows
        ]

    # Hydrate narrative summary from OpponentProfile.
    profile = session.execute(
        select(OpponentProfile).where(
            OpponentProfile.character_id == match.character_id,
            OpponentProfile.player_id == match.player_id,
        )
    ).scalar_one_or_none()

    return PostMatchStatus(
        match_id=match_id,
        status=analysis.status.value,
        steps_completed=list(analysis.steps_completed or []),
        error=analysis.error,
        elo_delta_applied=analysis.elo_delta_applied,
        floor_raised=bool(analysis.floor_raised),
        critical_moments=list(analysis.critical_moments or []),
        features=dict(analysis.features or {}),
        generated_memories=memory_snippets,
        narrative_summary=profile.narrative_summary if profile else None,
        started_at=analysis.started_at,
        completed_at=analysis.completed_at,
    )


@router.get("/{match_id}/moves", response_model=MoveList)
def list_moves(
    match_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(200, ge=1, le=500),
    player: Player = Depends(require_player),
    session: Session = Depends(get_session),
) -> MoveList:
    try:
        match = service.get_match(session, match_id)
    except MatchNotFound:
        raise HTTPException(status_code=404, detail="Match not found")
    if match.player_id != player.id:
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
