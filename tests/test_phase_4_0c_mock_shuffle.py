"""Phase 4.0c: regression test for the MockEngine knight/bishop shuffling bug.

Pre-fix behaviour: the mock engine picked the alphabetically-first UCI move
on every turn. In many positions that produced 2-move cycles (A-B-A-B) that
lasted for several plies in the opening and made the agent look broken.

These tests drive the mock engine through a multi-turn sequence from the
starting position and assert that no direct reversal of the engine's own
prior move is ever chosen when a non-reversal alternative exists.
"""

from __future__ import annotations

import chess

from app.engine.base import EngineConfig
from app.engine.mock_engine import (
    MockEngine,
    _is_reversal,
    _pick_non_shuffle,
    _recent_own_moves,
)


def _cfg() -> EngineConfig:
    return EngineConfig(engine_name="mock", target_elo=1200, time_budget_seconds=0.1)


# --- unit tests on the filters -------------------------------------------


def test_is_reversal_detects_direct_undo():
    m1 = chess.Move.from_uci("g1f3")
    m1_rev = chess.Move.from_uci("f3g1")
    assert _is_reversal(m1_rev, m1) is True


def test_is_reversal_false_for_unrelated_move():
    m1 = chess.Move.from_uci("g1f3")
    other = chess.Move.from_uci("e2e4")
    assert _is_reversal(other, m1) is False


def test_is_reversal_false_when_promotion_present():
    # Promotion moves cannot be reversed in the "undo" sense.
    prior = chess.Move.from_uci("e7e8q")
    candidate = chess.Move.from_uci("e8e7")
    assert _is_reversal(candidate, prior) is False


def test_recent_own_moves_walks_stack_in_pairs():
    b = chess.Board()
    # white Nf3, black Nc6, white Ng1, black Nb8
    for uci in ["g1f3", "b8c6", "f3g1", "c6b8"]:
        b.push_uci(uci)
    # It's now white to move again; own recent moves are g1f3 (oldest) and f3g1 (newest).
    recents = _recent_own_moves(b, depth=3)
    assert [m.uci() for m in recents] == ["f3g1", "g1f3"]


# --- integration: drive MockEngine for several plies ---------------------


def _play_n_agent_turns(n: int, agent_plays_white: bool = True) -> list[chess.Move]:
    """Simulate `n` of the agent's own turns. Opponent moves are deterministic
    but varied (not shuffling) so any shuffle must come from the agent.
    """
    eng = MockEngine()
    board = chess.Board()
    # A few scripted opponent moves — enough for a 6-ply agent sequence.
    opponent_script = [
        "e7e5", "d7d6", "c7c6", "b8a6", "a6b8", "f7f6", "g7g6", "h7h6", "b7b6",
    ]

    own_moves: list[chess.Move] = []
    if not agent_plays_white:
        # Opponent (white) opens.
        board.push_uci("e2e4")
    for i in range(n):
        result = eng.get_move(board, _cfg())
        mv = chess.Move.from_uci(result.move)
        own_moves.append(mv)
        board.push(mv)
        if not board.is_game_over() and i < len(opponent_script):
            # Opponent plays a legal move (fall back to first legal if the
            # scripted one isn't legal in this position).
            opp_uci = opponent_script[i]
            opp_mv = chess.Move.from_uci(opp_uci)
            if opp_mv not in board.legal_moves:
                opp_mv = sorted(board.legal_moves, key=lambda m: m.uci())[0]
            board.push(opp_mv)
    return own_moves


def test_mock_engine_does_not_two_cycle_shuffle_six_plies():
    moves = _play_n_agent_turns(6, agent_plays_white=True)
    # Previously: g1f3, f3g1, g1f3, f3g1, g1f3, f3g1. With fix: second move
    # must differ from the reverse of the first; third must differ from
    # reverse of the second; etc.
    for i in range(1, len(moves)):
        prev = moves[i - 1]
        cur = moves[i]
        assert not _is_reversal(cur, prev), (
            f"Move {i} ({cur.uci()}) reverses prior own move ({prev.uci()}); "
            f"sequence so far: {[m.uci() for m in moves[: i + 1]]}"
        )


def test_mock_engine_does_not_shuffle_as_black():
    moves = _play_n_agent_turns(6, agent_plays_white=False)
    for i in range(1, len(moves)):
        assert not _is_reversal(moves[i], moves[i - 1]), (
            f"Black agent shuffled at move {i}: "
            f"{[m.uci() for m in moves[: i + 1]]}"
        )


def test_pick_non_shuffle_falls_back_when_all_are_reversals():
    """Pathological: legal is only the reversal. Fall back to first."""
    b = chess.Board()
    b.push_uci("e2e4")  # white
    b.push_uci("e7e5")  # black
    b.push_uci("g1f3")  # white pushes our prior move
    b.push_uci("b8c6")  # black replies
    # It's white's turn again — reversal would be f3g1.
    reversal = chess.Move.from_uci("f3g1")
    # Force the "legal" list to contain only the reversal to exercise fallback.
    chosen = _pick_non_shuffle(b, [reversal])
    assert chosen == reversal


def test_mock_engine_plays_legal_moves():
    """Smoke test: nothing exotic like a promotion hiccup or illegal pick."""
    eng = MockEngine()
    board = chess.Board()
    for _ in range(20):
        if board.is_game_over():
            break
        r = eng.get_move(board, _cfg())
        mv = chess.Move.from_uci(r.move)
        assert mv in board.legal_moves
        board.push(mv)
        if board.is_game_over():
            break
        # Opponent also plays mock (symmetric). It will also avoid shuffles.
        r2 = eng.get_move(board, _cfg())
        board.push_uci(r2.move)
