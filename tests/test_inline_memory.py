"""Unit and integration tests for Block 4 inline memory (save_inline_memory + wiring)."""

from __future__ import annotations

import asyncio
import logging

import chess
import pytest
from pydantic import ValidationError
from sqlalchemy import func, select

from app.db import SessionLocal
from app.models.character import Character, CharacterState
from app.models.match import Color, Match, MatchStatus, Player
from app.models.memory import Memory, MemoryScope
from app.schemas.agents import InlineMemoryRequest, SoulResponse


# --- Helpers ----------------------------------------------------------------


def _mk_player(s, username: str) -> Player:
    p = Player(username=username, display_name=username)
    s.add(p)
    s.flush()
    return p


def _mk_character(s) -> Character:
    c = Character(
        name="TestChar", aggression=5, risk_tolerance=5, patience=5, trash_talk=5,
        target_elo=1400, current_elo=1400, floor_elo=1400, max_elo=1800,
        adaptive=True, state=CharacterState.READY,
    )
    s.add(c)
    s.flush()
    return c


def _mk_match(s, player: Player, character: Character, move_count: int = 0) -> Match:
    m = Match(
        character_id=character.id, player_id=player.id,
        player_color=Color.BLACK,
        status=MatchStatus.IN_PROGRESS,
        initial_fen=chess.STARTING_FEN, current_fen=chess.STARTING_FEN,
        move_count=move_count, character_elo_at_start=character.current_elo,
    )
    s.add(m)
    s.commit()
    s.refresh(m)
    return m


def _request(**overrides) -> InlineMemoryRequest:
    base = dict(
        narrative_text="This player told me they are rated 1850 on chess.com.",
        triggers=["1850", "chess.com", "rating"],
        relevance_tags=["rating", "strength"],
        emotional_valence=0.1,
        type="observation",
    )
    base.update(overrides)
    return InlineMemoryRequest(**base)


def _fake_embed(texts: list[str]) -> list[list[float]]:
    """Return a deterministic non-zero unit vector keyed by text content."""
    out = []
    for t in texts:
        vec = [0.0] * 384
        vec[abs(hash(t)) % 384] = 1.0
        out.append(vec)
    return out


def _memory_count(char_id: str, player_id: str | None = None) -> int:
    with SessionLocal() as s:
        q = select(func.count()).select_from(Memory).where(Memory.character_id == char_id)
        if player_id is not None:
            q = q.where(Memory.player_id == player_id)
        return s.execute(q).scalar_one()


# --- Unit tests: schema (tests 1-2) -----------------------------------------


def test_soul_response_save_memory_roundtrip():
    """`save_memory` field survives SoulResponse construction and is null by default."""
    from app.schemas.agents import MoodDeltas

    # Null by default
    sr = SoulResponse(mood_deltas=MoodDeltas())
    assert sr.save_memory is None

    # Populated when provided
    sr2 = SoulResponse(speak="Nice.", save_memory=_request(), mood_deltas=MoodDeltas())
    assert sr2.save_memory is not None
    assert sr2.save_memory.type == "observation"
    assert "1850" in sr2.save_memory.triggers


def test_inline_memory_request_rejects_short_narrative():
    """Narrative under 20 chars fails validation."""
    with pytest.raises(ValidationError):
        InlineMemoryRequest(
            narrative_text="too short",  # < 20 chars
            triggers=["a", "b"],
            relevance_tags=["tag"],
            emotional_valence=0.0,
            type="observation",
        )


# --- Unit tests: save_inline_memory (tests 3-6) -----------------------------


@pytest.mark.asyncio
async def test_save_inline_memory_persists_with_embedding(monkeypatch):
    """Happy path: memory is persisted with the embedding vector populated."""
    from app.memory import inline_save

    monkeypatch.setattr(inline_save, "embed_texts", _fake_embed)

    with SessionLocal() as s:
        player = _mk_player(s, "im_t1_p")
        char = _mk_character(s)
        player_id, char_id = player.id, char.id

    await inline_save.save_inline_memory(_request(), character_id=char_id, player_id=player_id)

    with SessionLocal() as s:
        rows = list(s.execute(
            select(Memory).where(Memory.character_id == char_id, Memory.player_id == player_id)
        ).scalars())

    assert len(rows) == 1
    m = rows[0]
    assert m.narrative_text == _request().narrative_text
    assert m.scope == MemoryScope.OPPONENT_SPECIFIC
    assert m.embedding is not None and len(m.embedding) == 384
    assert m.surface_count == 0
    assert m.last_surfaced_at is None
    assert m.match_id is None


@pytest.mark.asyncio
async def test_save_inline_memory_survives_embed_failure(monkeypatch, caplog):
    """If embed_texts raises, save_inline_memory logs WARNING and returns without raising."""
    from app.memory import inline_save

    def _boom(texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedding service down")

    monkeypatch.setattr(inline_save, "embed_texts", _boom)

    with SessionLocal() as s:
        player = _mk_player(s, "im_t2_p")
        char = _mk_character(s)
        player_id, char_id = player.id, char.id

    with caplog.at_level(logging.WARNING, logger="app.memory.inline_save"):
        # Must not raise.
        await inline_save.save_inline_memory(_request(), character_id=char_id, player_id=player_id)

    assert any("inline_memory_failed" in r.message for r in caplog.records)
    assert _memory_count(char_id) == 0


@pytest.mark.asyncio
async def test_dedup_same_memory_skipped(monkeypatch, caplog):
    """Saving the same memory twice: second call is skipped (cosine similarity = 1.0)."""
    from app.memory import inline_save

    fixed_vec = [0.0] * 384
    fixed_vec[0] = 1.0

    def _identical(texts: list[str]) -> list[list[float]]:
        return [list(fixed_vec) for _ in texts]

    monkeypatch.setattr(inline_save, "embed_texts", _identical)

    with SessionLocal() as s:
        player = _mk_player(s, "im_t3_p")
        char = _mk_character(s)
        player_id, char_id = player.id, char.id

    req = _request()
    with caplog.at_level(logging.INFO, logger="app.memory.inline_save"):
        await inline_save.save_inline_memory(req, character_id=char_id, player_id=player_id)
        await inline_save.save_inline_memory(req, character_id=char_id, player_id=player_id)

    assert _memory_count(char_id) == 1
    assert any("inline_memory_dedup_skipped" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_dedup_distinct_memories_both_persist(monkeypatch):
    """Saving two orthogonal memories (cosine sim = 0.0): both are persisted."""
    from app.memory import inline_save

    call_idx = 0

    def _orthogonal(texts: list[str]) -> list[list[float]]:
        nonlocal call_idx
        vecs = []
        for _ in texts:
            vec = [0.0] * 384
            vec[call_idx % 384] = 1.0
            call_idx += 1
            vecs.append(vec)
        return vecs

    monkeypatch.setattr(inline_save, "embed_texts", _orthogonal)

    with SessionLocal() as s:
        player = _mk_player(s, "im_t4_p")
        char = _mk_character(s)
        player_id, char_id = player.id, char.id

    req1 = _request()
    req2 = _request(
        narrative_text="This player exclusively plays the Catalan opening and has for years.",
        triggers=["Catalan", "opening", "preference"],
        relevance_tags=["opening", "preference"],
        emotional_valence=-0.1,
        type="habit",
    )

    await inline_save.save_inline_memory(req1, character_id=char_id, player_id=player_id)
    await inline_save.save_inline_memory(req2, character_id=char_id, player_id=player_id)

    assert _memory_count(char_id) == 2


# --- Integration tests (tests 5-7) ------------------------------------------


@pytest.mark.asyncio
async def test_room_pipeline_saves_memory_when_soul_requests(monkeypatch):
    """Pre-match: Soul returning save_memory → memory is persisted after pipeline."""
    from app.memory import inline_save as inline_save_mod
    from app.models.chat import CharacterChatSession, ChatSessionStatus
    from app.schemas.agents import MoodDeltas
    from app.sockets import room_server as room_mod

    monkeypatch.setattr(inline_save_mod, "embed_texts", _fake_embed)

    def _fake_soul(character, inp):
        return SoulResponse(speak="Solid rating.", save_memory=_request(), mood_deltas=MoodDeltas())

    def _fake_subconscious(session, character, inp, llm=None):
        return []

    monkeypatch.setattr(room_mod, "run_soul", _fake_soul)
    monkeypatch.setattr(room_mod, "run_subconscious", _fake_subconscious)

    with SessionLocal() as s:
        player = _mk_player(s, "im_room_p")
        char = _mk_character(s)
        cs = CharacterChatSession(
            character_id=char.id, player_id=player.id,
            status=ChatSessionStatus.ACTIVE, pending_notes=[],
        )
        s.add(cs)
        s.commit()
        s.refresh(cs)
        session_id, char_id, player_id = cs.id, char.id, player.id

    from app.sockets.room_server import run_room_agent_pipeline
    await run_room_agent_pipeline(
        chat_session_id=session_id,
        character_id=char_id,
        player_id=player_id,
        player_text="I'm rated 1850 on chess.com.",
    )
    await asyncio.sleep(0.5)  # allow create_task to run

    assert _memory_count(char_id, player_id) == 1


@pytest.mark.asyncio
async def test_in_match_pipeline_saves_memory_when_soul_requests(monkeypatch):
    """In-match: Soul returning save_memory → memory is persisted after engine pipeline."""
    from app.engine.registry import reset_engines_for_testing
    from app.matches import streaming as streaming_mod
    from app.matches.streaming import TurnEmitters, _run_engine_and_agents
    from app.memory import inline_save as inline_save_mod
    from app.redis_client import reset_memory_store_for_testing
    from app.schemas.agents import MoodDeltas

    monkeypatch.setattr(inline_save_mod, "embed_texts", _fake_embed)

    def _fake_soul(character, inp):
        return SoulResponse(speak="Noted.", save_memory=_request(), mood_deltas=MoodDeltas())

    def _fake_subconscious(session, character, inp, llm=None):
        return []

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
            player = _mk_player(s, "im_match_p")
            char = _mk_character(s)
            match = _mk_match(s, player, char, move_count=0)
            match_id, char_id, player_id = match.id, char.id, player.id

        from app.config import get_settings
        await _run_engine_and_agents(match_id=match_id, emitters=emitters, settings=get_settings())
        await asyncio.sleep(0.5)  # allow create_task to run
    finally:
        reset_memory_store_for_testing()
        reset_engines_for_testing()

    assert _memory_count(char_id, player_id) == 1


@pytest.mark.asyncio
async def test_tab_close_task_survives_disconnect(monkeypatch):
    """asyncio.create_task completes even if the initiating coroutine has returned.

    Simulates a client disconnecting immediately after Soul returns: the task
    continues to run because it holds no reference to the socket.
    """
    from app.memory import inline_save as inline_save_mod

    monkeypatch.setattr(inline_save_mod, "embed_texts", _fake_embed)

    with SessionLocal() as s:
        player = _mk_player(s, "im_tc_p")
        char = _mk_character(s)
        player_id, char_id = player.id, char.id

    # Schedule the task without awaiting it — the "client handler" returns here.
    # On a real server the socket disconnect fires at this point, but the task
    # has already been handed to the event loop and runs independently.
    task = asyncio.create_task(
        inline_save_mod.save_inline_memory(_request(), character_id=char_id, player_id=player_id)
    )

    # The key guarantee: awaiting the task (e.g., at graceful shutdown) succeeds.
    await task
    assert task.done() and task.exception() is None

    assert _memory_count(char_id) == 1
