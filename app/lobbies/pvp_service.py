"""Phase 4.2f — PvP match engine.

Create + drive player-vs-player matches inside a lobby. No engine, no
Soul, no memory, no post-match LLM steps — just chess rules, chess
clocks (deferred), and Elo.

Elo rules (anti-cheat):
- Public lobby → standard expected-score Elo applied on match end.
- Private lobby → `PvpMatch.is_private=True`; ratchet skipped, matching
  the guardrail locked into the Phase 4.0b memory.

Move storage: JSON list on PvpMatch.moves.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import chess
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.director.elo import (
    ELO_DELTA_CAP,
    FLOOR_RAISE_MARGIN,
    FLOOR_RAISE_STEP,
    FLOOR_RAISE_WINDOW,
    RatchetResult,
)
from app.lobbies.service import (
    LobbyClosed,
    LobbyError,
    LobbyForbidden,
    active_members,
)
from app.models.lobby import (
    Lobby,
    LobbyStatus,
    PvpMatch,
    PvpMatchResult,
    PvpMatchStatus,
)
from app.models.match import Player
from app.post_match.elo_apply import (
    K_ESTABLISHED,
    K_NEW,
    K_SWITCH_GAMES,
    expected_score,
)

logger = logging.getLogger(__name__)

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


# --- errors ---------------------------------------------------------------


class PvpError(LobbyError):
    code = "pvp_error"


class PvpMatchInProgress(PvpError):
    code = "pvp_match_in_progress"


class PvpMatchNotInProgress(PvpError):
    code = "pvp_match_not_in_progress"


class PvpNotYourTurn(PvpError):
    code = "pvp_not_your_turn"


class PvpIllegalMove(PvpError):
    code = "pvp_illegal_move"


class PvpMatchNotFound(PvpError):
    code = "pvp_match_not_found"


class PvpNotParticipant(PvpError):
    code = "pvp_not_participant"


# --- create ---------------------------------------------------------------


def start_match(
    session: Session, lobby: Lobby, *, by: Player, white_choice: str = "random"
) -> PvpMatch:
    """Host-only. Requires two seated members. Creates a fresh PvpMatch,
    picks sides (by `white_choice` ∈ {"white", "black", "random"} from
    the host's POV), flips the lobby into IN_MATCH.

    Locks `is_private` from the lobby at this moment — later door
    toggles don't retroactively change Elo treatment of this match.
    """
    if lobby.status == LobbyStatus.CLOSED:
        raise LobbyClosed()
    if lobby.status == LobbyStatus.IN_MATCH:
        raise PvpMatchInProgress()
    if lobby.host_id != by.id:
        raise LobbyForbidden("only the host can start the match")

    members = active_members(session, lobby.id)
    if len(members) != 2:
        raise PvpError("need exactly 2 players to start")

    host_mem = next((m for m in members if m.player_id == lobby.host_id), None)
    guest_mem = next((m for m in members if m.player_id != lobby.host_id), None)
    if host_mem is None or guest_mem is None:
        raise PvpError("host or guest missing from membership roster")

    # Resolve side preference.
    choice = (white_choice or "random").lower()
    if choice not in ("white", "black", "random"):
        choice = "random"
    if choice == "random":
        host_plays_white = random.random() < 0.5
    else:
        host_plays_white = choice == "white"

    white_id = lobby.host_id if host_plays_white else guest_mem.player_id
    black_id = guest_mem.player_id if host_plays_white else lobby.host_id

    white_player = session.get(Player, white_id)
    black_player = session.get(Player, black_id)
    if white_player is None or black_player is None:
        raise PvpError("seat player missing")

    match = PvpMatch(
        lobby_id=lobby.id,
        white_player_id=white_id,
        black_player_id=black_id,
        initial_fen=START_FEN,
        current_fen=START_FEN,
        moves=[],
        move_count=0,
        status=PvpMatchStatus.IN_PROGRESS,
        is_private=bool(lobby.is_private),
        white_elo_at_start=int(white_player.elo),
        black_elo_at_start=int(black_player.elo),
        extra_state={},
    )
    session.add(match)
    session.flush()

    lobby.status = LobbyStatus.IN_MATCH
    lobby.current_match_id = match.id
    lobby.updated_at = datetime.utcnow()
    session.commit()
    session.refresh(match)
    return match


# --- move -----------------------------------------------------------------


@dataclass(frozen=True)
class AppliedPvpMove:
    move: dict[str, Any]         # the JSON-shaped move row appended to .moves
    fen_after: str
    game_over: bool
    result: PvpMatchResult | None
    reason: str | None           # "checkmate" | "stalemate" | "threefold_repetition" | "fifty_move" | "insufficient_material" | None


def apply_move(
    session: Session,
    match: PvpMatch,
    *,
    by: Player,
    uci: str,
) -> AppliedPvpMove:
    """Validate + apply `uci` as `by`'s move. Persists the move to
    `match.moves`, updates `current_fen`, flips status + result if the
    game ended.
    """
    if match.status != PvpMatchStatus.IN_PROGRESS:
        raise PvpMatchNotInProgress()

    # Identify mover's color.
    if by.id == match.white_player_id:
        my_side = chess.WHITE
        side_name = "white"
    elif by.id == match.black_player_id:
        my_side = chess.BLACK
        side_name = "black"
    else:
        raise PvpNotParticipant()

    board = chess.Board(match.current_fen)
    if board.turn != my_side:
        raise PvpNotYourTurn()

    try:
        move = chess.Move.from_uci(uci)
    except ValueError:
        raise PvpIllegalMove(f"malformed uci: {uci}")
    if move not in board.legal_moves:
        raise PvpIllegalMove(f"not a legal move: {uci}")

    san = board.san(move)
    board.push(move)
    fen_after = board.fen()
    now_iso = datetime.utcnow().isoformat() + "Z"

    entry = {
        "move_number": match.move_count + 1,
        "uci": uci,
        "san": san,
        "fen_after": fen_after,
        "side": side_name,
        "ts": now_iso,
    }
    # SQLAlchemy JSON columns don't auto-detect in-place list mutation.
    # Reassign the list so the change is flushed.
    match.moves = (match.moves or []) + [entry]
    match.move_count = len(match.moves)
    match.current_fen = fen_after

    # Terminal?
    ended = False
    result: PvpMatchResult | None = None
    reason: str | None = None
    if board.is_checkmate():
        ended = True
        reason = "checkmate"
        # After push(), the side to move is the one in check with no legal
        # escape — they're the loser. So the mover just won.
        result = PvpMatchResult.WHITE_WIN if my_side == chess.WHITE else PvpMatchResult.BLACK_WIN
    elif board.is_stalemate():
        ended = True; reason = "stalemate"; result = PvpMatchResult.DRAW
    elif board.is_insufficient_material():
        ended = True; reason = "insufficient_material"; result = PvpMatchResult.DRAW
    elif board.is_fivefold_repetition() or board.can_claim_threefold_repetition():
        ended = True; reason = "threefold_repetition"; result = PvpMatchResult.DRAW
    elif board.is_seventyfive_moves() or board.can_claim_fifty_moves():
        ended = True; reason = "fifty_move"; result = PvpMatchResult.DRAW

    if ended:
        _finalize(session, match, status=PvpMatchStatus.COMPLETED, result=result, reason=reason)
    else:
        session.commit()

    return AppliedPvpMove(
        move=entry,
        fen_after=fen_after,
        game_over=ended,
        result=result,
        reason=reason,
    )


# --- resign + abandon ----------------------------------------------------


def resign(session: Session, match: PvpMatch, *, by: Player) -> PvpMatchResult:
    if match.status != PvpMatchStatus.IN_PROGRESS:
        raise PvpMatchNotInProgress()
    if by.id not in (match.white_player_id, match.black_player_id):
        raise PvpNotParticipant()

    # Resigning side loses; the other wins.
    if by.id == match.white_player_id:
        result = PvpMatchResult.BLACK_WIN
    else:
        result = PvpMatchResult.WHITE_WIN

    _finalize(session, match, status=PvpMatchStatus.RESIGNED, result=result, reason="resigned")
    return result


def abandon_for_disconnect(
    session: Session, match: PvpMatch, *, abandoning_player_id: str
) -> PvpMatchResult:
    """Called when the disconnect cooldown on `abandoning_player_id`
    expires. Opposite player wins. Elo applies unless the match is private.
    """
    if match.status != PvpMatchStatus.IN_PROGRESS:
        raise PvpMatchNotInProgress()
    if abandoning_player_id == match.white_player_id:
        result = PvpMatchResult.BLACK_WIN
    elif abandoning_player_id == match.black_player_id:
        result = PvpMatchResult.WHITE_WIN
    else:
        raise PvpNotParticipant()
    _finalize(session, match, status=PvpMatchStatus.ABANDONED, result=result, reason="abandoned")
    return result


# --- elo on match end -----------------------------------------------------


def _finalize(
    session: Session,
    match: PvpMatch,
    *,
    status: PvpMatchStatus,
    result: PvpMatchResult | None,
    reason: str | None,
) -> None:
    """Stamp terminal state, apply Elo (if public), and return lobby
    to OPEN so the players can rematch.
    """
    match.status = status
    match.result = result
    match.ended_at = datetime.utcnow()
    match.extra_state = {**(match.extra_state or {}), "reason": reason}

    if not match.is_private and result is not None and result != PvpMatchResult.ABANDONED:
        _apply_pvp_elo(session, match)
    elif match.is_private:
        logger.info("[pvp] match %s is_private → skipping Elo ratchet", match.id)

    # Flip lobby back to OPEN so the same pair can rematch.
    lobby = session.get(Lobby, match.lobby_id)
    if lobby is not None and lobby.status == LobbyStatus.IN_MATCH:
        lobby.status = LobbyStatus.OPEN
        lobby.current_match_id = None
        lobby.updated_at = datetime.utcnow()

    session.commit()


def _k_factor(games_played: int) -> int:
    return K_NEW if games_played < K_SWITCH_GAMES else K_ESTABLISHED


def _pvp_games_played(session: Session, player_id: str) -> int:
    from sqlalchemy import func

    return int(
        session.execute(
            select(func.count(PvpMatch.id)).where(
                (PvpMatch.white_player_id == player_id)
                | (PvpMatch.black_player_id == player_id)
            )
        ).scalar() or 0
    )


def _apply_pvp_elo(session: Session, match: PvpMatch) -> None:
    """Standard expected-score delta, applied to both players. No ACPL
    (no engine analysis on PvP games), no easy-win cap either — PvP
    players self-select opponents via matchmaking already, which the
    band system smooths. Short-game scaling retained to avoid reward
    for instant resigns.
    """
    white = session.get(Player, match.white_player_id)
    black = session.get(Player, match.black_player_id)
    if white is None or black is None:
        logger.warning("[pvp] elo apply: player gone for match %s", match.id)
        return

    w_elo = int(white.elo)
    b_elo = int(black.elo)
    e_white = expected_score(w_elo, b_elo)
    e_black = 1.0 - e_white

    if match.result == PvpMatchResult.WHITE_WIN:
        actual_white, actual_black = 1.0, 0.0
    elif match.result == PvpMatchResult.BLACK_WIN:
        actual_white, actual_black = 0.0, 1.0
    else:
        actual_white = actual_black = 0.5  # draw

    k_white = _k_factor(_pvp_games_played(session, white.id))
    k_black = _k_factor(_pvp_games_played(session, black.id))

    raw_white = k_white * (actual_white - e_white)
    raw_black = k_black * (actual_black - e_black)

    # Short-game scaling. Mirrors compute_elo_delta's rule.
    plies = match.move_count or 0
    if plies < 2:
        raw_white = max(-3.0, min(3.0, raw_white))
        raw_black = max(-3.0, min(3.0, raw_black))
    elif plies < 10:
        raw_white *= 0.3
        raw_black *= 0.3

    white_result = _player_ratchet(session, white, raw_white)
    black_result = _player_ratchet(session, black, raw_black)

    white.elo = white_result.new_current_elo
    white.elo_floor = white_result.new_floor_elo
    black.elo = black_result.new_current_elo
    black.elo_floor = black_result.new_floor_elo

    match.white_elo_at_end = white_result.new_current_elo
    match.black_elo_at_end = black_result.new_current_elo

    logger.info(
        "[pvp-elo] match=%s white=%s (%s → %s, Δ%s) black=%s (%s → %s, Δ%s) "
        "e_white=%.3f actual_white=%.1f",
        match.id,
        white.id, w_elo, white_result.new_current_elo, white_result.current_elo_change,
        black.id, b_elo, black_result.new_current_elo, black_result.current_elo_change,
        e_white, actual_white,
    )


def _player_ratchet(session: Session, player: Player, raw_delta: float) -> RatchetResult:
    """Trimmed copy of `_apply_player_ratchet` from elo_apply — pulls the
    recent window straight from PvpMatch (the PvE version looks at the
    `matches` table).
    """
    from sqlalchemy import or_

    rows = session.execute(
        select(PvpMatch)
        .where(
            or_(
                PvpMatch.white_player_id == player.id,
                PvpMatch.black_player_id == player.id,
            )
        )
        .where(PvpMatch.ended_at.is_not(None))
        .order_by(PvpMatch.ended_at.desc())
        .limit(FLOOR_RAISE_WINDOW - 1)
    ).scalars()
    recent = []
    for m in rows:
        end = (
            m.white_elo_at_end if m.white_player_id == player.id else m.black_elo_at_end
        )
        if end is not None:
            recent.append(end)
    recent.reverse()

    change = int(round(raw_delta))
    change = max(-ELO_DELTA_CAP, min(ELO_DELTA_CAP, change))
    new_current = max(player.elo_floor, min(player.elo_ceiling, player.elo + change))
    actual_change = new_current - player.elo

    window = recent[-(FLOOR_RAISE_WINDOW - 1):] + [new_current]
    floor_raised = False
    new_floor = player.elo_floor
    if len(window) == FLOOR_RAISE_WINDOW and all(
        c >= player.elo_floor + FLOOR_RAISE_MARGIN for c in window
    ):
        new_floor = min(player.elo_ceiling, player.elo_floor + FLOOR_RAISE_STEP)
        floor_raised = new_floor != player.elo_floor

    return RatchetResult(
        new_current_elo=new_current,
        new_floor_elo=new_floor,
        current_elo_change=actual_change,
        floor_elo_raised=floor_raised,
    )


# --- lookup ---------------------------------------------------------------


def get_match(session: Session, match_id: str) -> PvpMatch:
    m = session.get(PvpMatch, match_id)
    if m is None:
        raise PvpMatchNotFound()
    return m
