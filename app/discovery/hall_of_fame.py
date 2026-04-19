"""Per-character hall of fame: top N players vs. one character, by wins."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from app.models.match import Color, Match, MatchResult, MatchStatus, Player


@dataclass(frozen=True)
class HallOfFameRow:
    rank: int
    player_id: str
    username: str
    display_name: str
    wins: int        # player wins vs. this character
    total_matches: int
    win_rate: float


def hall_of_fame_for_character(
    session: Session, *, character_id: str, limit: int = 10
) -> list[HallOfFameRow]:
    total = func.count(Match.id)
    player_wins = func.sum(
        case(
            (Match.status == MatchStatus.ABANDONED, 0),
            (
                (Match.player_color == Color.WHITE)
                & (Match.result == MatchResult.WHITE_WIN),
                1,
            ),
            (
                (Match.player_color == Color.BLACK)
                & (Match.result == MatchResult.BLACK_WIN),
                1,
            ),
            else_=0,
        )
    )
    stmt = (
        select(
            Player.id, Player.username, Player.display_name,
            total.label("total"), player_wins.label("wins"),
        )
        .join(Match, Match.player_id == Player.id)
        .where(
            Match.character_id == character_id,
            Match.status.in_([MatchStatus.COMPLETED, MatchStatus.ABANDONED]),
            Player.username != "legacy_system",
        )
        .group_by(Player.id, Player.username, Player.display_name)
    )
    rows = session.execute(stmt).all()
    # Sort: wins desc, then total matches desc, then win rate desc.
    sorted_rows = sorted(
        rows,
        key=lambda r: (r.wins or 0, r.total or 0, (r.wins or 0) / (r.total or 1)),
        reverse=True,
    )
    top = sorted_rows[:limit]
    out: list[HallOfFameRow] = []
    for i, r in enumerate(top, start=1):
        wins = int(r.wins or 0)
        total = int(r.total or 0)
        out.append(
            HallOfFameRow(
                rank=i,
                player_id=r.id,
                username=r.username,
                display_name=r.display_name,
                wins=wins,
                total_matches=total,
                win_rate=(wins / total) if total else 0.0,
            )
        )
    return out
