"""Match lifecycle + turn loop.

Async API because the engine step is blocking and we don't want to stall
FastAPI's event loop. All DB reads/writes happen on the calling thread —
the only thing we ship to a worker thread is the engine's `get_move`.
"""

from __future__ import annotations

import asyncio
import logging
import random
from datetime import datetime

import chess
from sqlalchemy.orm import Session

from app.director import MoodState, choose_engine_config, initial_mood_for_character, smooth_mood
from app.director.director import MatchContext
from app.director.mood import load_mood, save_mood
from app.engine import EngineUnavailable, available_engines, get_engine
from app.engine.base import EngineConfig, MoveResult
from app.models.character import Character
from app.models.match import (
    Color,
    Match,
    MatchResult,
    MatchStatus,
    Move,
    Player,
)

logger = logging.getLogger(__name__)

START_FEN = chess.STARTING_FEN


class MatchError(RuntimeError):
    pass


class MatchNotFound(MatchError):
    pass


class IllegalMove(MatchError):
    pass


class NotYourTurn(MatchError):
    pass


class GameAlreadyOver(MatchError):
    pass


# --- Helpers ------------------------------------------------------------


def _phase_from_board(board: chess.Board) -> str:
    # Keep this in sync with board_abstraction._phase; duplicated because
    # the matcher service shouldn't depend on board_abstraction.
    if board.fullmove_number < 10:
        return "opening"
    non_pawn_material = 0
    for piece in board.piece_map().values():
        if piece.piece_type in (chess.PAWN, chess.KING):
            continue
        non_pawn_material += {chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}[
            piece.piece_type
        ]
    return "endgame" if non_pawn_material <= 16 else "middlegame"


def _board_from_match(match: Match) -> chess.Board:
    return chess.Board(match.current_fen)


def _char_color(match: Match) -> chess.Color:
    return chess.BLACK if match.player_color == Color.WHITE else chess.WHITE


def _finalize_if_over(match: Match, board: chess.Board) -> None:
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return
    match.status = MatchStatus.COMPLETED
    match.ended_at = datetime.utcnow()
    match.character_elo_at_end = match.character_elo_at_start  # 2a: no ratchet during live play
    if outcome.winner is None:
        match.result = MatchResult.DRAW
    elif outcome.winner == chess.WHITE:
        match.result = MatchResult.WHITE_WIN
    else:
        match.result = MatchResult.BLACK_WIN


def _load_or_init_mood(session: Session, match: Match) -> tuple[MoodState, MoodState]:
    """Return (raw, smoothed) mood. Initialized from character if missing."""
    smoothed = load_mood(match.id, smoothed=True)
    raw = load_mood(match.id, smoothed=False)
    if smoothed is None or raw is None:
        character = session.get(Character, match.character_id)
        if character is None:
            raise MatchError(f"Character {match.character_id} missing")
        initial = initial_mood_for_character(character)
        save_mood(match.id, initial, smoothed=True)
        save_mood(match.id, initial, smoothed=False)
        return initial, initial
    return raw, smoothed


def _apply_mood_step(match_id: str, raw: MoodState, previous_smoothed: MoodState) -> MoodState:
    smoothed = smooth_mood(previous_smoothed, raw)
    save_mood(match_id, smoothed, smoothed=True)
    save_mood(match_id, raw, smoothed=False)
    return smoothed


# --- Creation / lookup --------------------------------------------------


def get_or_create_player(session: Session, *, player_id: str | None, display_name: str = "Guest") -> Player:
    if player_id:
        existing = session.get(Player, player_id)
        if existing is not None:
            return existing
    player = Player(display_name=display_name or "Guest")
    session.add(player)
    session.flush()
    return player


def create_match(
    session: Session,
    *,
    character_id: str,
    player_id: str,
    player_color: str = "random",
) -> Match:
    character = session.get(Character, character_id)
    if character is None or character.deleted_at is not None:
        raise MatchError(f"Character {character_id} not found")
    player = session.get(Player, player_id)
    if player is None:
        raise MatchError(f"Player {player_id} not found")

    if player_color == "random":
        chosen = random.choice([Color.WHITE, Color.BLACK])
    elif player_color == "white":
        chosen = Color.WHITE
    elif player_color == "black":
        chosen = Color.BLACK
    else:
        raise MatchError(f"Invalid player_color: {player_color}")

    match = Match(
        character_id=character.id,
        player_id=player.id,
        player_color=chosen,
        status=MatchStatus.IN_PROGRESS,
        initial_fen=START_FEN,
        current_fen=START_FEN,
        move_count=0,
        character_elo_at_start=character.current_elo,
    )
    session.add(match)
    session.flush()

    # Seed mood so the first engine turn can read it.
    _load_or_init_mood(session, match)
    return match


def get_match(session: Session, match_id: str) -> Match:
    match = session.get(Match, match_id)
    if match is None:
        raise MatchNotFound(match_id)
    return match


# --- Turn loop ----------------------------------------------------------


def _persist_move(
    session: Session,
    *,
    match: Match,
    board_before: chess.Board,
    engine_result: MoveResult | None,
    uci: str,
    side: chess.Color,
    player_chat_before: str | None = None,
    mood_snapshot: dict | None = None,
) -> Move:
    move = chess.Move.from_uci(uci)
    if move not in board_before.legal_moves:
        raise IllegalMove(f"Illegal move: {uci}")
    san = board_before.san(move)
    board_after = board_before.copy()
    board_after.push(move)

    move_row = Move(
        match_id=match.id,
        move_number=match.move_count + 1,
        side=Color.WHITE if side == chess.WHITE else Color.BLACK,
        uci=uci,
        san=san,
        fen_after=board_after.fen(),
        engine_name=engine_result.engine_name if engine_result else None,
        time_taken_ms=engine_result.time_taken_ms if engine_result else None,
        eval_cp=engine_result.eval_cp if engine_result else None,
        considered_moves=[cm.model_dump() for cm in engine_result.considered_moves]
        if engine_result
        else [],
        thinking_depth=engine_result.thinking_depth if engine_result else None,
        player_chat_before=player_chat_before,
        mood_snapshot=mood_snapshot or {},
    )
    session.add(move_row)

    match.current_fen = board_after.fen()
    match.move_count += 1
    _finalize_if_over(match, board_after)

    session.flush()
    return move_row


async def _engine_turn(session: Session, match: Match) -> Move:
    board = _board_from_match(match)
    character = session.get(Character, match.character_id)
    if character is None:
        raise MatchError("Character missing mid-match")

    raw, smoothed = _load_or_init_mood(session, match)
    # 2a: mood doesn't evolve from chat (no Soul yet). Still run the smoothing
    # step so the Director reads the persisted smoothed value.
    smoothed = _apply_mood_step(match.id, raw, smoothed)

    available = frozenset(available_engines())
    if not available:
        raise EngineUnavailable("No engines available — install Maia-2 or Stockfish")

    context = MatchContext(
        move_number=board.fullmove_number,
        game_phase=_phase_from_board(board),
        player_color=match.player_color.value,
        engines_available=available,
    )
    config: EngineConfig = choose_engine_config(
        character=character, mood=smoothed, match_context=context
    )

    engine = get_engine(config.engine_name)

    def _run() -> MoveResult:
        return engine.get_move(board, config)

    result = await asyncio.to_thread(_run)

    return _persist_move(
        session,
        match=match,
        board_before=board,
        engine_result=result,
        uci=result.move,
        side=board.turn,
        mood_snapshot=smoothed.to_dict(),
    )


async def start_match_play(session: Session, match: Match) -> Move | None:
    """If the character is white, play the opening move before returning to the user."""
    if match.status != MatchStatus.IN_PROGRESS:
        return None
    board = _board_from_match(match)
    if board.turn != _char_color(match):
        return None
    return await _engine_turn(session, match)


async def apply_player_move(
    session: Session,
    *,
    match_id: str,
    uci: str,
) -> tuple[Move, Move | None]:
    match = get_match(session, match_id)
    if match.status != MatchStatus.IN_PROGRESS:
        raise GameAlreadyOver(match.status.value)

    board = _board_from_match(match)
    player_color = chess.WHITE if match.player_color == Color.WHITE else chess.BLACK
    if board.turn != player_color:
        raise NotYourTurn("It's not your turn")

    player_move = _persist_move(
        session,
        match=match,
        board_before=board,
        engine_result=None,
        uci=uci,
        side=player_color,
    )

    if match.status != MatchStatus.IN_PROGRESS:
        return player_move, None

    engine_move = await _engine_turn(session, match)
    return player_move, engine_move


def resign(session: Session, *, match_id: str) -> Match:
    match = get_match(session, match_id)
    if match.status != MatchStatus.IN_PROGRESS:
        raise GameAlreadyOver(match.status.value)
    match.status = MatchStatus.ABANDONED
    match.result = MatchResult.ABANDONED
    match.ended_at = datetime.utcnow()
    match.character_elo_at_end = match.character_elo_at_start
    session.flush()
    return match


# --- Outcome helpers for the API ---------------------------------------


def player_outcome(match: Match) -> str | None:
    if match.status == MatchStatus.IN_PROGRESS:
        return None
    if match.result == MatchResult.DRAW:
        return "draw"
    if match.result == MatchResult.ABANDONED:
        return "resigned"
    player_side = match.player_color
    winning_side = Color.WHITE if match.result == MatchResult.WHITE_WIN else Color.BLACK
    return "win" if player_side == winning_side else "loss"
