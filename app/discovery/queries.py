"""Discovery page backends: live + recent match listings.

Single visibility filter (`visible_character_filter`) is re-used by every
discovery/leaderboard surface so the content-rating rules stay consistent.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import ColumnElement, or_, select
from sqlalchemy.orm import Session

from app.models.character import Character, ContentRating, Visibility, rating_level
from app.models.match import Match, MatchResult, MatchStatus, Player


@dataclass(frozen=True)
class MatchSummary:
    match_id: str
    character_id: str
    character_name: str
    character_avatar: str
    character_content_rating: str
    character_visibility: str
    player_id: str
    player_username: str
    status: str
    result: str | None
    move_count: int
    player_color: str
    started_at: datetime
    ended_at: datetime | None


def visible_character_filter(viewer: Player) -> list[ColumnElement[bool]]:
    """SQLAlchemy filter clauses for characters the viewer is allowed to see.

    Mirrors `app.web.routes._visible_filter` — extracted here so both the
    discovery endpoints and the leaderboard queries share one definition.
    """
    max_idx = rating_level(viewer.max_content_rating)
    allowed = [
        r for r in (ContentRating.FAMILY, ContentRating.MATURE, ContentRating.UNRESTRICTED)
        if rating_level(r) <= max_idx
    ]
    return [
        Character.deleted_at.is_(None),
        Character.content_rating.in_(allowed),
        or_(Character.visibility == Visibility.PUBLIC, Character.owner_id == viewer.id),
    ]


def _row_to_summary(match: Match, character: Character, player: Player) -> MatchSummary:
    return MatchSummary(
        match_id=match.id,
        character_id=character.id,
        character_name=character.name,
        character_avatar=character.avatar_emoji or "♟",
        character_content_rating=character.content_rating.value
        if hasattr(character.content_rating, "value") else str(character.content_rating),
        character_visibility=character.visibility.value
        if hasattr(character.visibility, "value") else str(character.visibility),
        player_id=player.id,
        player_username=player.username,
        status=match.status.value if hasattr(match.status, "value") else str(match.status),
        result=match.result.value if match.result and hasattr(match.result, "value")
        else (str(match.result) if match.result else None),
        move_count=match.move_count,
        player_color=match.player_color.value
        if hasattr(match.player_color, "value") else str(match.player_color),
        started_at=match.started_at,
        ended_at=match.ended_at,
    )


def list_live_matches(
    session: Session,
    *,
    viewer: Player,
    limit: int = 20,
    offset: int = 0,
    character_id: str | None = None,
) -> list[MatchSummary]:
    """In-progress matches the viewer can see, most-recently started first.

    Excludes the viewer's own matches — they're already in them.
    If character_id is given, restrict to matches against that character.
    """
    clauses = [
        Match.status == MatchStatus.IN_PROGRESS,
        Match.player_id != viewer.id,
        *visible_character_filter(viewer),
    ]
    if character_id is not None:
        clauses.append(Match.character_id == character_id)
    stmt = (
        select(Match, Character, Player)
        .join(Character, Match.character_id == Character.id)
        .join(Player, Match.player_id == Player.id)
        .where(*clauses)
        .order_by(Match.started_at.desc())
        .offset(offset)
        .limit(limit)
    )
    return [_row_to_summary(m, c, p) for m, c, p in session.execute(stmt).all()]


def list_recent_matches(
    session: Session,
    *,
    viewer: Player,
    limit: int = 20,
    offset: int = 0,
    character_id: str | None = None,
) -> list[MatchSummary]:
    """Completed or abandoned matches the viewer can see, most-recently ended first.

    Includes the viewer's own matches so they can navigate back to a summary.
    If character_id is given, restrict to matches against that character.
    """
    clauses = [
        Match.status.in_([MatchStatus.COMPLETED, MatchStatus.RESIGNED, MatchStatus.ABANDONED]),
        Match.ended_at.is_not(None),
        *visible_character_filter(viewer),
    ]
    if character_id is not None:
        clauses.append(Match.character_id == character_id)
    stmt = (
        select(Match, Character, Player)
        .join(Character, Match.character_id == Character.id)
        .join(Player, Match.player_id == Player.id)
        .where(*clauses)
        .order_by(Match.ended_at.desc())
        .offset(offset)
        .limit(limit)
    )
    return [_row_to_summary(m, c, p) for m, c, p in session.execute(stmt).all()]
