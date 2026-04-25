"""Block 3: per-session concurrency lock tests.

Three test groups:
  1. Locks module — pure async, no DB required.
  2. Greeting lock — two simultaneous connects yield exactly one greeting turn.
  3. Opening move lock — two simultaneous connects yield exactly one engine move.

Pipeline serialization is covered indirectly by the lock-module tests; the
lock itself is the invariant, not the pipeline internals.
"""

from __future__ import annotations

import asyncio

import chess
import pytest

from app.db import SessionLocal
from app.models.character import Character, CharacterState
from app.models.chat import CharacterChatSession, ChatSessionStatus
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
        aggression=5,
        risk_tolerance=5,
        patience=5,
        trash_talk=5,
        target_elo=1400,
        current_elo=1400,
        floor_elo=1400,
        max_elo=1800,
        adaptive=True,
        state=CharacterState.READY,
    )
    s.add(c)
    s.flush()
    return c


def _mk_chat_session(s, player: Player, character: Character) -> CharacterChatSession:
    cs = CharacterChatSession(
        character_id=character.id,
        player_id=player.id,
        status=ChatSessionStatus.ACTIVE,
        pending_notes=[],
    )
    s.add(cs)
    s.commit()
    s.refresh(cs)
    return cs


def _mk_match(s, player: Player, character: Character, move_count: int = 0) -> Match:
    m = Match(
        character_id=character.id,
        player_id=player.id,
        player_color=Color.BLACK,  # character plays white
        status=MatchStatus.IN_PROGRESS,
        initial_fen=chess.STARTING_FEN,
        current_fen=chess.STARTING_FEN,
        move_count=move_count,
        character_elo_at_start=character.current_elo,
    )
    s.add(m)
    s.commit()
    s.refresh(m)
    return m


# --- 1. Locks module --------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_session_lock_serializes():
    from app.concurrency.locks import _chat_session_locks, chat_session_lock

    key = "test-serial-chat-xyz"
    _chat_session_locks.pop(key, None)

    order: list[str] = []

    async def task_a():
        async with chat_session_lock(key):
            order.append("a-start")
            await asyncio.sleep(0.05)
            order.append("a-end")

    async def task_b():
        await asyncio.sleep(0.01)  # let a acquire first
        async with chat_session_lock(key):
            order.append("b-start")

    await asyncio.gather(task_a(), task_b())
    assert order == ["a-start", "a-end", "b-start"]


@pytest.mark.asyncio
async def test_match_lock_serializes():
    from app.concurrency.locks import _match_locks, match_lock

    key = "test-serial-match-abc"
    _match_locks.pop(key, None)

    order: list[str] = []

    async def task_a():
        async with match_lock(key):
            order.append("a")
            await asyncio.sleep(0.05)

    async def task_b():
        await asyncio.sleep(0.01)
        async with match_lock(key):
            order.append("b")

    await asyncio.gather(task_a(), task_b())
    assert order == ["a", "b"]


@pytest.mark.asyncio
async def test_lock_releases_on_exception():
    from app.concurrency.locks import _chat_session_locks, chat_session_lock

    key = "test-exc-release"
    _chat_session_locks.pop(key, None)

    with pytest.raises(RuntimeError):
        async with chat_session_lock(key):
            raise RuntimeError("boom")

    # Lock must be released — next acquisition must not block.
    acquired = False
    async with chat_session_lock(key):
        acquired = True
    assert acquired


@pytest.mark.asyncio
async def test_different_keys_do_not_block_each_other():
    from app.concurrency.locks import _chat_session_locks, chat_session_lock

    for k in ("key-a", "key-b"):
        _chat_session_locks.pop(k, None)

    order: list[str] = []

    async def task_a():
        async with chat_session_lock("key-a"):
            order.append("a-start")
            await asyncio.sleep(0.05)
            order.append("a-end")

    async def task_b():
        await asyncio.sleep(0.01)
        async with chat_session_lock("key-b"):
            order.append("b-start")

    await asyncio.gather(task_a(), task_b())
    # b-start should appear while a is still sleeping (different keys → no blocking)
    assert order.index("b-start") < order.index("a-end")


# --- 2. Greeting lock -------------------------------------------------------


@pytest.mark.asyncio
async def test_greeting_lock_prevents_double_greeting(monkeypatch):
    """Two concurrent _locked_fire_greeting tasks on an empty session
    yield exactly one greeting turn in the DB."""
    from app.sockets import room_server

    fire_count = 0

    async def _mock_fire_greeting(session_id: str, char_id: str, player_id: str) -> None:
        nonlocal fire_count
        fire_count += 1
        await asyncio.sleep(0.05)  # simulate LLM latency
        # Write a greeting turn so the second task's re-check sees it.
        with SessionLocal() as s:
            from app.characters.chat_service import _next_turn_number
            from app.models.chat import CharacterChatTurn, ChatTurnRole

            cs = s.get(CharacterChatSession, session_id)
            if cs is None:
                return
            turn = CharacterChatTurn(
                session_id=session_id,
                turn_number=_next_turn_number(s, cs),
                role=ChatTurnRole.CHARACTER,
                text="Hello!",
            )
            s.add(turn)
            s.commit()

    monkeypatch.setattr(room_server, "_fire_greeting", _mock_fire_greeting)

    with SessionLocal() as s:
        player = _mk_player(s, f"greet_p_{id(s) % 9999}")
        char = _mk_character(s)
        cs = _mk_chat_session(s, player, char)
        session_id = cs.id
        char_id = char.id
        player_id = player.id

    # Simulate two simultaneous connects.
    await asyncio.gather(
        room_server._locked_fire_greeting(session_id, char_id, player_id),
        room_server._locked_fire_greeting(session_id, char_id, player_id),
    )

    with SessionLocal() as s:
        from app.characters.chat_service import get_turns

        cs = s.get(CharacterChatSession, session_id)
        turns = get_turns(s, cs)

    assert len(turns) == 1, f"Expected 1 greeting turn, got {len(turns)}"
    assert fire_count == 1, f"_fire_greeting called {fire_count} times, expected 1"


@pytest.mark.asyncio
async def test_greeting_lock_skips_when_already_greeted(monkeypatch):
    """If a greeting turn already exists when _locked_fire_greeting runs,
    it must not call _fire_greeting at all."""
    from app.sockets import room_server

    fire_count = 0

    async def _mock_fire_greeting(*_args) -> None:
        nonlocal fire_count
        fire_count += 1

    monkeypatch.setattr(room_server, "_fire_greeting", _mock_fire_greeting)

    with SessionLocal() as s:
        player = _mk_player(s, f"greet_skip_{id(s) % 9999}")
        char = _mk_character(s)
        cs = _mk_chat_session(s, player, char)
        # Pre-seed a greeting turn so the session is not empty.
        from app.characters.chat_service import _next_turn_number
        from app.models.chat import CharacterChatTurn, ChatTurnRole

        turn = CharacterChatTurn(
            session_id=cs.id,
            turn_number=_next_turn_number(s, cs),
            role=ChatTurnRole.CHARACTER,
            text="Already here!",
        )
        s.add(turn)
        s.commit()
        session_id = cs.id
        char_id = char.id
        player_id = player.id

    await room_server._locked_fire_greeting(session_id, char_id, player_id)

    assert fire_count == 0


# --- 3. Pipeline lock -------------------------------------------------------


@pytest.mark.asyncio
async def test_run_pipeline_locked_serializes(monkeypatch):
    """Two concurrent _run_pipeline_locked calls on the same session
    execute the inner pipeline sequentially, not concurrently."""
    from app.sockets import room_server

    order: list[str] = []

    async def _mock_pipeline(
        *, chat_session_id, character_id, player_id, player_text
    ) -> None:
        order.append(f"start-{player_text}")
        await asyncio.sleep(0.05)
        order.append(f"end-{player_text}")

    monkeypatch.setattr(room_server, "run_room_agent_pipeline", _mock_pipeline)

    with SessionLocal() as s:
        player = _mk_player(s, f"pipe_p_{id(s) % 9999}")
        char = _mk_character(s)
        cs = _mk_chat_session(s, player, char)
        session_id = cs.id
        char_id = char.id
        player_id = player.id

    await asyncio.gather(
        room_server._run_pipeline_locked(
            chat_session_id=session_id,
            character_id=char_id,
            player_id=player_id,
            player_text="first",
        ),
        room_server._run_pipeline_locked(
            chat_session_id=session_id,
            character_id=char_id,
            player_id=player_id,
            player_text="second",
        ),
    )

    # Both ran — but end-first must precede start-second.
    assert "end-first" in order and "start-second" in order
    assert order.index("end-first") < order.index("start-second"), (
        f"Pipelines ran concurrently: {order}"
    )


# --- 4. Opening-move lock ---------------------------------------------------


@pytest.mark.asyncio
async def test_opening_move_lock_prevents_double_fire(monkeypatch):
    """Two near-simultaneous _fire_opening_move calls on a fresh match
    result in exactly one engine invocation."""
    from app.matches.streaming import TurnEmitters
    from app.sockets import server as sio_server

    engine_count = 0

    async def _mock_engine(*, match_id, emitters, settings) -> None:
        nonlocal engine_count
        engine_count += 1
        await asyncio.sleep(0.05)
        with SessionLocal() as s:
            m = s.get(Match, match_id)
            if m is not None:
                m.move_count = 1
                s.commit()

    async def _noop(*_a, **_kw) -> None:
        pass

    noop_emitters = TurnEmitters(
        on_player_move=_noop,
        on_thinking=_noop,
        on_memory_surfaced=_noop,
        on_agent_move=_noop,
        on_agent_chat=_noop,
        on_mood_update=_noop,
        on_match_ended=_noop,
        on_post_match_kickoff=_noop,
    )

    monkeypatch.setattr(
        "app.matches.streaming._run_engine_and_agents", _mock_engine
    )
    monkeypatch.setattr(sio_server, "_build_turn_emitters", lambda _mid: noop_emitters)

    with SessionLocal() as s:
        player = _mk_player(s, f"open_p_{id(s) % 9999}")
        char = _mk_character(s)
        match = _mk_match(s, player, char, move_count=0)
        match_id = match.id

    await asyncio.gather(
        sio_server._fire_opening_move(match_id),
        sio_server._fire_opening_move(match_id),
    )

    assert engine_count == 1, f"Engine called {engine_count} times, expected 1"

    with SessionLocal() as s:
        m = s.get(Match, match_id)
    assert m.move_count == 1
