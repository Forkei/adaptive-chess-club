"""Unit and integration tests for app.engine.diversity_guard."""

from __future__ import annotations

import chess
import pytest

from app.db import SessionLocal
from app.engine.base import ConsideredMove
from app.engine.diversity_guard import _is_shuffle, filter_shuffle_moves
from app.models.character import Character, CharacterState
from app.models.match import Color, Match, MatchStatus, Player


# --- Helpers ----------------------------------------------------------------


def _mk_player(s, username: str) -> Player:
    p = Player(username=username, display_name=username)
    s.add(p)
    s.flush()
    return p


def _mk_character(s) -> Character:
    c = Character(
        name="TestChar",
        aggression=5, risk_tolerance=5, patience=5, trash_talk=5,
        target_elo=1400, current_elo=1400, floor_elo=1400, max_elo=1800,
        adaptive=True, state=CharacterState.READY,
    )
    s.add(c)
    s.flush()
    return c


def _mk_match(s, player: Player, character: Character, move_count: int = 0) -> Match:
    m = Match(
        character_id=character.id, player_id=player.id,
        player_color=Color.BLACK,  # character plays white
        status=MatchStatus.IN_PROGRESS,
        initial_fen=chess.STARTING_FEN, current_fen=chess.STARTING_FEN,
        move_count=move_count, character_elo_at_start=character.current_elo,
    )
    s.add(m)
    s.commit()
    s.refresh(m)
    return m


# --- Unit tests: filter_shuffle_moves / _is_shuffle -------------------------


def test_empty_recent_never_shuffles():
    """With no recent moves the top candidate is always returned unchanged."""
    board = chess.Board()
    candidates = [
        ConsideredMove(uci="e2e4", san="e4"),
        ConsideredMove(uci="d2d4", san="d4"),
    ]
    result = filter_shuffle_moves(candidates, [], board)
    assert result.uci == "e2e4"


def test_top_candidate_not_shuffle_returned():
    """Top move is not a shuffle — returned as-is."""
    # After 1. Nf3 Nc6: white to move, knight on f3, recent own move was g1f3.
    board = chess.Board("r1bqkbnr/pppppppp/2n5/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 2 2")
    recent = [chess.Move.from_uci("g1f3")]
    candidates = [
        ConsideredMove(uci="e2e4", san="e4"),   # not a shuffle
        ConsideredMove(uci="f3g1", san="Ng1"),  # would be a shuffle
    ]
    result = filter_shuffle_moves(candidates, recent, board)
    assert result.uci == "e2e4"


def test_top_shuffle_returns_second_candidate():
    """Top move is a shuffle (knight back to g1) → second candidate chosen."""
    board = chess.Board("r1bqkbnr/pppppppp/2n5/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 2 2")
    recent = [chess.Move.from_uci("g1f3")]
    candidates = [
        ConsideredMove(uci="f3g1", san="Ng1"),  # shuffle
        ConsideredMove(uci="e2e4", san="e4"),   # not a shuffle
    ]
    result = filter_shuffle_moves(candidates, recent, board)
    assert result.uci == "e2e4"


def test_all_shuffles_fallback_to_first():
    """All candidates are shuffles → first candidate returned as fallback."""
    # Position with knight on f3 and bishop on g5; white to move.
    board = chess.Board("r1bqkbnr/pppppp1p/2n3p1/6B1/8/5N2/PPPPPPPP/RN1QKB1R w KQkq - 0 3")
    recent = [
        chess.Move.from_uci("c1g5"),  # bishop recently left c1
        chess.Move.from_uci("g1f3"),  # knight recently left g1
    ]
    candidates = [
        ConsideredMove(uci="f3g1", san="Ng1"),  # shuffle: knight back to g1
        ConsideredMove(uci="g5c1", san="Bc1"),  # shuffle: bishop back to c1
    ]
    result = filter_shuffle_moves(candidates, recent, board)
    assert result.uci == "f3g1"  # fallback: first candidate


def test_different_piece_type_not_shuffle():
    """Candidate targets a recently vacated square but is a different piece type — not flagged."""
    # Rook on e4, Knight on f3, King on h1. A piece (of rook type) recently came from e1.
    board = chess.Board("8/8/8/8/4R3/5N2/8/7K w - - 0 1")
    recent = [chess.Move.from_uci("e1e4")]  # rook-type piece moved from e1 to e4
    candidates = [
        ConsideredMove(uci="f3e1", san="Ne1"),  # knight → e1: different type than rook
        ConsideredMove(uci="f3d4", san="Nd4"),
    ]
    result = filter_shuffle_moves(candidates, recent, board)
    assert result.uci == "f3e1"  # not flagged as a shuffle


def test_extract_own_recent_moves_ordering():
    """_extract_own_recent_moves returns the side-to-move's moves, newest first."""
    from app.matches.streaming import _extract_own_recent_moves

    board = chess.Board()
    board.push(chess.Move.from_uci("e2e4"))  # white
    board.push(chess.Move.from_uci("e7e5"))  # black
    board.push(chess.Move.from_uci("g1f3"))  # white
    board.push(chess.Move.from_uci("b8c6"))  # black — now white to move

    recent = _extract_own_recent_moves(board, lookback=2)
    assert recent == [chess.Move.from_uci("g1f3"), chess.Move.from_uci("e2e4")]

    assert _extract_own_recent_moves(board, lookback=1) == [chess.Move.from_uci("g1f3")]
    assert _extract_own_recent_moves(board, lookback=0) == []


# --- Integration test -------------------------------------------------------


@pytest.mark.asyncio
async def test_filter_wired_into_pipeline(monkeypatch):
    """filter_shuffle_moves is called inside _run_engine_and_agents_inner."""
    from app.engine.registry import reset_engines_for_testing
    from app.matches import streaming as streaming_mod
    from app.matches.streaming import TurnEmitters, _run_engine_and_agents
    from app.redis_client import reset_memory_store_for_testing
    from app.schemas.agents import MoodDeltas, SoulResponse

    filter_calls: list[int] = []

    def _recording_filter(candidates, own_recent_moves, board, lookback_plies=6):
        filter_calls.append(len(candidates))
        return candidates[0]

    def _fake_soul(character, inp):
        return SoulResponse(speak=None, emotion="neutral", emotion_intensity=0.3, mood_deltas=MoodDeltas())

    def _fake_subconscious(session, character, inp, llm=None):
        return []

    monkeypatch.setattr(streaming_mod, "filter_shuffle_moves", _recording_filter)
    monkeypatch.setattr(streaming_mod, "run_soul", _fake_soul)
    monkeypatch.setattr(streaming_mod, "run_subconscious", _fake_subconscious)

    async def _noop(*a, **kw):
        pass

    emitters = TurnEmitters(
        on_player_move=_noop, on_thinking=_noop, on_memory_surfaced=_noop,
        on_agent_move=_noop, on_agent_chat=_noop, on_mood_update=_noop,
        on_match_ended=_noop, on_post_match_kickoff=_noop,
    )

    reset_engines_for_testing()
    reset_memory_store_for_testing()
    try:
        with SessionLocal() as s:
            player = _mk_player(s, f"dg_p_{id(s) % 9999}")
            char = _mk_character(s)
            match = _mk_match(s, player, char, move_count=0)
            match_id = match.id

        from app.config import get_settings
        await _run_engine_and_agents(match_id=match_id, emitters=emitters, settings=get_settings())
    finally:
        reset_memory_store_for_testing()
        reset_engines_for_testing()

    assert len(filter_calls) == 1, f"filter_shuffle_moves should be called once, got {len(filter_calls)}"
