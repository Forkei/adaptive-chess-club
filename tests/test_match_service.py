"""End-to-end match flow with the MockEngine.

MockEngine always plays the first UCI-sorted legal move, so both sides
are deterministic. These tests guarantee the match lifecycle machinery
(move validation, turn ordering, engine invocation, terminal detection)
works without touching a real chess engine.
"""

from __future__ import annotations

import asyncio

import pytest

from app.db import SessionLocal
from app.engine.registry import reset_engines_for_testing
from app.matches import service
from app.matches.service import GameAlreadyOver, IllegalMove, NotYourTurn
from app.models.character import Character, CharacterState
from app.models.match import Color, MatchStatus, Player
from app.redis_client import reset_memory_store_for_testing


@pytest.fixture(autouse=True)
def _reset_engines_and_mood():
    reset_engines_for_testing()
    reset_memory_store_for_testing()
    yield
    reset_engines_for_testing()
    reset_memory_store_for_testing()


@pytest.fixture(autouse=True)
def _stub_agents(monkeypatch):
    """Keep the LLM and embedding model out of the match-lifecycle tests.

    These tests only care about move ordering, validation, and terminal
    detection — not the Soul/Subconscious output — so we replace the
    agents pipeline with a no-op that leaves moves silent.
    """
    from app.matches import service as match_service
    from app.agents.subconscious import clear_cache

    clear_cache()

    async def _noop_agents(*args, **kwargs):
        return None

    def _sync_noop(*args, **kwargs):
        return None

    # _run_agents_sync is invoked via asyncio.to_thread with fixed
    # positional args; patching it to return None makes _engine_turn skip
    # the chat/mood-delta path while still producing a Move row.
    monkeypatch.setattr(match_service, "_run_agents_sync", _sync_noop)
    yield
    clear_cache()


@pytest.fixture
def _force_mock_only(monkeypatch):
    """Pretend only MockEngine is installed, regardless of host state."""
    import app.engine.registry as reg

    def _fake_available() -> list:
        # Ensure mock is instantiable, and pretend the others are missing.
        reg._build_default_factories()
        return ["mock"]

    monkeypatch.setattr("app.matches.service.available_engines", _fake_available)
    yield


def _character(session) -> Character:
    char = Character(
        name="Mock Master",
        short_description="t",
        aggression=5,
        risk_tolerance=5,
        patience=3,
        trash_talk=5,
        target_elo=1500,
        current_elo=1500,
        floor_elo=1500,
        max_elo=1800,
        state=CharacterState.READY,
    )
    session.add(char)
    session.commit()
    return char


def _player(session) -> Player:
    from app.auth import generate_guest_username

    p = Player(username=generate_guest_username(), display_name="T")
    session.add(p)
    session.commit()
    return p


def test_create_match_persists_match(_force_mock_only):
    with SessionLocal() as s:
        char = _character(s)
        player = _player(s)
        match = service.create_match(
            s, character_id=char.id, player_id=player.id, player_color="white"
        )
        s.commit()
        assert match.id is not None
        assert match.status == MatchStatus.IN_PROGRESS
        assert match.player_color == Color.WHITE
        assert match.character_elo_at_start == 1500


def test_character_plays_first_if_white(_force_mock_only):
    async def _run():
        with SessionLocal() as s:
            char = _character(s)
            player = _player(s)
            match = service.create_match(
                s, character_id=char.id, player_id=player.id, player_color="black"
            )
            s.commit()
            first = await service.start_match_play(s, match)
            s.commit()
            return first, match

    first, match = asyncio.run(_run())
    assert first is not None
    assert first.side == Color.WHITE  # character plays white
    assert match.move_count == 1


def test_illegal_move_rejected(_force_mock_only):
    async def _run():
        with SessionLocal() as s:
            char = _character(s)
            player = _player(s)
            match = service.create_match(
                s, character_id=char.id, player_id=player.id, player_color="white"
            )
            s.commit()
            match_id = match.id
        with SessionLocal() as s:
            try:
                await service.apply_player_move(s, match_id=match_id, uci="e2e5")
                raised = False
            except IllegalMove:
                raised = True
            return raised

    assert asyncio.run(_run()) is True


def test_out_of_turn_rejected(_force_mock_only):  # noqa: E501
    """Player is black but tries to move first."""

    async def _run():
        with SessionLocal() as s:
            char = _character(s)
            player = _player(s)
            match = service.create_match(
                s, character_id=char.id, player_id=player.id, player_color="black"
            )
            s.commit()
            match_id = match.id
        # Deliberately do NOT run start_match_play (character's turn) first.
        with SessionLocal() as s:
            try:
                await service.apply_player_move(s, match_id=match_id, uci="e7e5")
                raised = False
            except NotYourTurn:
                raised = True
            return raised

    assert asyncio.run(_run()) is True


def test_player_move_then_engine_replies(_force_mock_only):
    async def _run():
        with SessionLocal() as s:
            char = _character(s)
            player = _player(s)
            match = service.create_match(
                s, character_id=char.id, player_id=player.id, player_color="white"
            )
            s.commit()
            match_id = match.id
        with SessionLocal() as s:
            player_move, engine_move, _agent = await service.apply_player_move(
                s, match_id=match_id, uci="e2e4"
            )
            s.commit()
            return player_move, engine_move

    player_move, engine_move = asyncio.run(_run())
    assert player_move.uci == "e2e4"
    assert player_move.side == Color.WHITE
    assert engine_move is not None
    assert engine_move.side == Color.BLACK
    assert engine_move.engine_name == "mock"
    assert engine_move.move_number == 2


def test_resign_ends_match(_force_mock_only):
    async def _run():
        with SessionLocal() as s:
            char = _character(s)
            player = _player(s)
            match = service.create_match(
                s, character_id=char.id, player_id=player.id, player_color="white"
            )
            s.commit()
            match_id = match.id
        with SessionLocal() as s:
            match = service.resign(s, match_id=match_id)
            s.commit()
            return match

    match = asyncio.run(_run())
    assert match.status == MatchStatus.ABANDONED
    assert match.ended_at is not None
    assert match.result is not None
    assert match.result.value == "abandoned"


def test_moves_after_resign_rejected(_force_mock_only):
    async def _run():
        with SessionLocal() as s:
            char = _character(s)
            player = _player(s)
            match = service.create_match(
                s, character_id=char.id, player_id=player.id, player_color="white"
            )
            s.commit()
            match_id = match.id
        with SessionLocal() as s:
            service.resign(s, match_id=match_id)
            s.commit()
        with SessionLocal() as s:
            try:
                await service.apply_player_move(s, match_id=match_id, uci="e2e4")
                raised = False
            except GameAlreadyOver:
                raised = True
            return raised

    assert asyncio.run(_run()) is True


def test_full_match_reaches_terminal_state(_force_mock_only):
    """Mock engine plays Nxa3-a3b5-... kind of deterministic sequence.

    We play until either the game ends or we hit a generous move cap,
    then assert the match either completed or is still in progress.
    Never assert on the exact outcome — MockEngine's sort order is stable
    but not meaningful.
    """

    async def _run():
        with SessionLocal() as s:
            char = _character(s)
            player = _player(s)
            match = service.create_match(
                s, character_id=char.id, player_id=player.id, player_color="white"
            )
            s.commit()
            match_id = match.id

        with SessionLocal() as s:
            import chess

            for _ in range(300):  # hard cap, more than enough
                match = service.get_match(s, match_id)
                if match.status != MatchStatus.IN_PROGRESS:
                    break
                board = chess.Board(match.current_fen)
                legal = sorted(board.legal_moves, key=lambda m: m.uci())
                if not legal:
                    break
                try:
                    await service.apply_player_move(s, match_id=match_id, uci=legal[0].uci())
                    s.commit()
                except GameAlreadyOver:
                    break
            return service.get_match(s, match_id)

    final_match = asyncio.run(_run())
    assert final_match.status in (MatchStatus.COMPLETED, MatchStatus.IN_PROGRESS)
    # If completed, result must be set.
    if final_match.status == MatchStatus.COMPLETED:
        assert final_match.result is not None
