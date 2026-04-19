"""Leaderboard aggregations — character + player variants.

Both leaderboards apply the same content-rating/visibility filter so a
family-rated viewer never sees mature-rated matches aggregated into any row.

**Abandoned matches count as character wins** (confirmed decision — 3c memo):
a character who provokes rage-quits consistently should reflect that in their
record. Don't filter ABANDONED out.

Within-window min-match threshold: rows with < `MIN_MATCHES_FOR_LEADERBOARD`
matches in the selected time window are excluded. Prevents "100% from 1 game"
noise. Configurable constant at module scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from sqlalchemy import Integer, case, func, select
from sqlalchemy.orm import Session

from app.discovery.queries import visible_character_filter
from app.models.character import Character
from app.models.match import Color, Match, MatchResult, MatchStatus, Player

MIN_MATCHES_FOR_LEADERBOARD = 5

LeaderboardWindow = Literal["all", "30d", "7d"]


def window_cutoff(window: LeaderboardWindow) -> datetime | None:
    """Return the `Match.ended_at` cutoff for the given window, or None for 'all'."""
    if window == "7d":
        return datetime.utcnow() - timedelta(days=7)
    if window == "30d":
        return datetime.utcnow() - timedelta(days=30)
    return None


@dataclass(frozen=True)
class CharacterLeaderboardRow:
    rank: int
    character_id: str
    character_name: str
    character_avatar: str
    current_elo: int
    total_matches: int
    wins: int        # character wins (incl. abandoned)
    losses: int
    draws: int
    win_rate: float  # 0.0 – 1.0


@dataclass(frozen=True)
class PlayerLeaderboardRow:
    rank: int
    player_id: str
    username: str
    display_name: str
    elo: int         # Patch Pass 2 Item 2: player Elo is the primary ranking axis
    total_matches: int
    wins: int        # player wins
    losses: int      # player losses (incl. own abandoned matches = char wins)
    draws: int
    win_rate: float


def _char_win_expr() -> tuple:
    """SQL case expressions for (character_wins, character_losses, draws) aggregations.

    Character is BLACK when `player_color == 'white'`, and vice versa.
    Abandoned matches ⇒ character wins (player resigned / timed out).
    """
    char_wins = func.sum(
        case(
            (Match.status == MatchStatus.ABANDONED, 1),
            (
                (Match.player_color == Color.WHITE)
                & (Match.result == MatchResult.BLACK_WIN),
                1,
            ),
            (
                (Match.player_color == Color.BLACK)
                & (Match.result == MatchResult.WHITE_WIN),
                1,
            ),
            else_=0,
        )
    )
    char_losses = func.sum(
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
    draws = func.sum(case((Match.result == MatchResult.DRAW, 1), else_=0))
    return char_wins, char_losses, draws


def _window_clauses(window: LeaderboardWindow) -> list:
    cutoff = window_cutoff(window)
    clauses = [
        Match.status.in_([MatchStatus.COMPLETED, MatchStatus.RESIGNED, MatchStatus.ABANDONED]),
        Match.ended_at.is_not(None),
    ]
    if cutoff is not None:
        clauses.append(Match.ended_at >= cutoff)
    return clauses


def character_leaderboard(
    session: Session,
    *,
    viewer: Player,
    window: LeaderboardWindow = "all",
) -> list[CharacterLeaderboardRow]:
    char_wins, char_losses, draws = _char_win_expr()
    total = func.count(Match.id)

    stmt = (
        select(
            Character.id,
            Character.name,
            Character.avatar_emoji,
            Character.current_elo,
            total.label("total"),
            char_wins.label("wins"),
            char_losses.label("losses"),
            draws.label("draws"),
        )
        .join(Match, Match.character_id == Character.id)
        .where(
            *_window_clauses(window),
            *visible_character_filter(viewer),
        )
        .group_by(Character.id, Character.name, Character.avatar_emoji, Character.current_elo)
        .having(total >= MIN_MATCHES_FOR_LEADERBOARD)
    )
    rows = session.execute(stmt).all()

    # Rank in Python (SQL doesn't need to sort — the row count is tiny).
    ranked = sorted(
        rows,
        key=lambda r: ((r.wins or 0) / (r.total or 1), r.total or 0),
        reverse=True,
    )
    out: list[CharacterLeaderboardRow] = []
    for i, r in enumerate(ranked, start=1):
        total_i = int(r.total or 0)
        wins_i = int(r.wins or 0)
        losses_i = int(r.losses or 0)
        draws_i = int(r.draws or 0)
        win_rate = (wins_i / total_i) if total_i else 0.0
        out.append(
            CharacterLeaderboardRow(
                rank=i,
                character_id=r.id,
                character_name=r.name,
                character_avatar=r.avatar_emoji or "♟",
                current_elo=int(r.current_elo),
                total_matches=total_i,
                wins=wins_i,
                losses=losses_i,
                draws=draws_i,
                win_rate=win_rate,
            )
        )
    return out


def player_leaderboard(
    session: Session,
    *,
    viewer: Player,
    window: LeaderboardWindow = "all",
) -> list[PlayerLeaderboardRow]:
    """Rank players by win rate vs. characters.

    A match's outcome is inverted from `_char_win_expr`: the player wins when
    the character loses, etc. Abandoned matches are player losses.
    """
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
    player_losses = func.sum(
        case(
            (Match.status == MatchStatus.ABANDONED, 1),
            (
                (Match.player_color == Color.WHITE)
                & (Match.result == MatchResult.BLACK_WIN),
                1,
            ),
            (
                (Match.player_color == Color.BLACK)
                & (Match.result == MatchResult.WHITE_WIN),
                1,
            ),
            else_=0,
        )
    )
    draws = func.sum(case((Match.result == MatchResult.DRAW, 1), else_=0))

    stmt = (
        select(
            Player.id,
            Player.username,
            Player.display_name,
            Player.elo,
            total.label("total"),
            player_wins.label("wins"),
            player_losses.label("losses"),
            draws.label("draws"),
        )
        .join(Match, Match.player_id == Player.id)
        .join(Character, Character.id == Match.character_id)
        .where(
            *_window_clauses(window),
            *visible_character_filter(viewer),
            Player.username != "legacy_system",  # never include the system fallback
        )
        .group_by(Player.id, Player.username, Player.display_name, Player.elo)
        .having(total >= MIN_MATCHES_FOR_LEADERBOARD)
    )
    rows = session.execute(stmt).all()
    # Patch Pass 2 Item 2: sort by Elo first, then total_matches as a tiebreaker.
    ranked = sorted(
        rows,
        key=lambda r: (int(r.elo or 0), r.total or 0),
        reverse=True,
    )
    out: list[PlayerLeaderboardRow] = []
    for i, r in enumerate(ranked, start=1):
        total_i = int(r.total or 0)
        wins_i = int(r.wins or 0)
        losses_i = int(r.losses or 0)
        draws_i = int(r.draws or 0)
        win_rate = (wins_i / total_i) if total_i else 0.0
        out.append(
            PlayerLeaderboardRow(
                rank=i,
                player_id=r.id,
                username=r.username,
                display_name=r.display_name,
                elo=int(r.elo or 1200),
                total_matches=total_i,
                wins=wins_i,
                losses=losses_i,
                draws=draws_i,
                win_rate=win_rate,
            )
        )
    return out
