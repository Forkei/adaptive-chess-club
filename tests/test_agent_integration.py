"""Integration test: Soul mood deltas flow through the match service.

We run a couple of player→engine turns with a stub Soul that reports
known mood_deltas, then verify raw mood moves in the expected direction
and smoothed mood trails behind (exponential smoothing with tau=3).
"""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from app.db import SessionLocal
from app.director.mood import MoodState, load_mood
from app.engine.registry import reset_engines_for_testing
from app.matches import service
from app.models.character import Character, CharacterState
from app.models.match import Player
from app.redis_client import reset_memory_store_for_testing
from app.schemas.agents import MoodDeltas, SoulResponse


@pytest.fixture(autouse=True)
def _reset_engines_and_store():
    reset_engines_for_testing()
    reset_memory_store_for_testing()
    yield
    reset_engines_for_testing()
    reset_memory_store_for_testing()


@pytest.fixture
def _mock_only(monkeypatch):
    import app.engine.registry as reg

    def _fake_available() -> list:
        reg._build_default_factories()
        return ["mock"]

    monkeypatch.setattr("app.matches.service.available_engines", _fake_available)
    yield


@pytest.fixture
def _stub_agents(monkeypatch):
    """Stub the agents pipeline to emit a known `confidence +0.1` delta
    every engine turn so we can verify raw/smoothed mood evolution."""

    from app.matches import service as match_service

    from app.agents.soul import SoulResponse as _SR
    from app.schemas.agents import MoodDeltas as _MD

    def _stub_sync(session, match, character, board_after, engine_result, smoothed, raw):
        resp = _SR(
            speak=None,
            emotion="focused",
            emotion_intensity=0.3,
            mood_deltas=_MD(confidence=0.1),
            referenced_memory_ids=[],
        )
        # Apply delta ourselves (mirroring real _run_agents_sync behavior).
        from app.director.mood import apply_deltas, save_mood, smooth_mood

        new_raw = apply_deltas(raw, resp.mood_deltas.to_dict())
        new_smoothed = smooth_mood(smoothed, new_raw)
        save_mood(match.id, new_raw, smoothed=False)
        save_mood(match.id, new_smoothed, smoothed=True)

        return match_service.AgentTurnOutcome(surfaced=[], soul=resp)

    monkeypatch.setattr(match_service, "_run_agents_sync", _stub_sync)
    yield


def _character(session) -> Character:
    char = Character(
        name="Deltas",
        short_description="t",
        aggression=5,
        risk_tolerance=5,
        patience=5,
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


def test_soul_mood_delta_applied_to_raw_and_smoothed(_mock_only, _stub_agents):
    async def _run():
        with SessionLocal() as s:
            char = _character(s)
            player = _player(s)
            match = service.create_match(
                s, character_id=char.id, player_id=player.id, player_color="white"
            )
            s.commit()
            match_id = match.id

        # Capture starting raw/smoothed confidence.
        raw_before = load_mood(match_id, smoothed=False)
        smoothed_before = load_mood(match_id, smoothed=True)
        assert raw_before is not None and smoothed_before is not None
        assert abs(raw_before.confidence - 0.5) < 1e-9
        assert abs(smoothed_before.confidence - 0.5) < 1e-9

        # Play three player→engine pairs.
        moves = ["e2e4", "d2d4", "g1f3"]
        for uci in moves:
            with SessionLocal() as s:
                await service.apply_player_move(s, match_id=match_id, uci=uci)
                s.commit()

        raw_after = load_mood(match_id, smoothed=False)
        smoothed_after = load_mood(match_id, smoothed=True)
        return raw_before, smoothed_before, raw_after, smoothed_after

    raw_before, smoothed_before, raw_after, smoothed_after = asyncio.run(_run())

    # Each engine turn adds +0.1 to raw confidence. 3 turns → +0.3 (capped at 1.0).
    assert raw_after.confidence > raw_before.confidence + 0.25
    # Smoothed trails raw: should be between raw_before and raw_after.
    assert smoothed_before.confidence < smoothed_after.confidence < raw_after.confidence


def test_soul_speak_and_memory_ids_land_on_move_row(_mock_only, monkeypatch):
    """When the Soul speaks, the chat + surfaced ids are persisted on the engine move."""
    from app.matches import service as match_service
    from app.agents.soul import SoulResponse as _SR
    from app.schemas.agents import MoodDeltas as _MD, SurfacedMemory

    async def _run():
        surfaced = [
            SurfacedMemory(
                memory_id="mem-stub",
                narrative_text="hello",
                triggers=[],
                relevance_tags=[],
                emotional_valence=0.0,
                scope="character_lore",
                score=0.5,
                retrieval_reason="stub",
            )
        ]

        def _stub_sync(session, match, character, board_after, engine_result, smoothed, raw):
            resp = _SR(
                speak="Your move was interesting.",
                emotion="pleased",
                emotion_intensity=0.5,
                mood_deltas=_MD(),
                referenced_memory_ids=["mem-stub"],
            )
            return match_service.AgentTurnOutcome(surfaced=surfaced, soul=resp)

        monkeypatch.setattr(match_service, "_run_agents_sync", _stub_sync)

        with SessionLocal() as s:
            char = _character(s)
            player = _player(s)
            match = service.create_match(
                s, character_id=char.id, player_id=player.id, player_color="white"
            )
            s.commit()
            match_id = match.id

        with SessionLocal() as s:
            player_move, engine_move, outcome = await service.apply_player_move(
                s, match_id=match_id, uci="e2e4"
            )
            s.commit()
            return engine_move, outcome

    engine_move, outcome = asyncio.run(_run())
    assert engine_move.agent_chat_after == "Your move was interesting."
    assert engine_move.surfaced_memory_ids == ["mem-stub"]
    assert outcome is not None
    assert outcome.soul.emotion == "pleased"
    assert len(outcome.surfaced) == 1


def test_opponent_note_queued_on_match(_mock_only, monkeypatch):
    """`note_about_opponent` ends up in match.extra_state.pending_opponent_notes."""
    from app.matches import service as match_service
    from app.agents.soul import SoulResponse as _SR
    from app.schemas.agents import MoodDeltas as _MD

    async def _run():
        def _stub_sync(session, match, character, board_after, engine_result, smoothed, raw):
            resp = _SR(
                speak=None,
                emotion="neutral",
                emotion_intensity=0.1,
                mood_deltas=_MD(),
                note_about_opponent="plays fast in open positions",
                referenced_memory_ids=[],
            )
            return match_service.AgentTurnOutcome(surfaced=[], soul=resp)

        monkeypatch.setattr(match_service, "_run_agents_sync", _stub_sync)

        with SessionLocal() as s:
            char = _character(s)
            player = _player(s)
            match = service.create_match(
                s, character_id=char.id, player_id=player.id, player_color="white"
            )
            s.commit()
            match_id = match.id
        with SessionLocal() as s:
            await service.apply_player_move(s, match_id=match_id, uci="e2e4")
            s.commit()
        with SessionLocal() as s:
            return service.get_match(s, match_id)

    match = asyncio.run(_run())
    notes = (match.extra_state or {}).get("pending_opponent_notes", [])
    assert notes
    assert notes[0]["note"] == "plays fast in open positions"
