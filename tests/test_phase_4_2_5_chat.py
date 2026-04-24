"""Phase 4.2.5 — pre-match chat service tests.

The Soul LLM is mocked out so tests are deterministic and free.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from app.characters import chat_service
from app.db import SessionLocal
from app.models.character import (
    Character,
    CharacterState,
    ContentRating,
    Visibility,
)
from app.models.chat import (
    CharacterChatSession,
    ChatSessionStatus,
    ChatTurnRole,
)
from app.models.match import Player
from app.schemas.agents import SoulResponse


def _mk_character(s) -> Character:
    c = Character(
        name="Test Character",
        short_description="short",
        backstory="A rich backstory for testing.",
        voice_descriptor="voice",
        target_elo=1400,
        current_elo=1400,
        floor_elo=1400,
        max_elo=1400,
        adaptive=False,
        is_preset=False,
        owner_id=None,
        state=CharacterState.READY,
        visibility=Visibility.PUBLIC,
        content_rating=ContentRating.FAMILY,
    )
    s.add(c)
    s.commit()
    s.refresh(c)
    return c


def _mk_player(s, username: str = "chat_p") -> Player:
    p = Player(username=username, display_name=username, elo=1200)
    s.add(p)
    s.commit()
    s.refresh(p)
    return p


def _soul_says(
    text: str, *, emotion: str = "neutral", intensity: float = 0.3,
    game_action: str = "none",
) -> SoulResponse:
    return SoulResponse(
        speak=text,
        emotion=emotion,
        emotion_intensity=intensity,
        game_action=game_action,
    )


# --- session lifecycle ---------------------------------------------------


def test_get_or_create_session_is_idempotent():
    with SessionLocal() as s:
        char = _mk_character(s)
        player = _mk_player(s, "ses_p")
        s1 = chat_service.get_or_create_session(s, character=char, player=player)
        s2 = chat_service.get_or_create_session(s, character=char, player=player)
        assert s1.id == s2.id


def test_close_session_marks_abandoned():
    with SessionLocal() as s:
        char = _mk_character(s)
        player = _mk_player(s, "close_p")
        sess = chat_service.get_or_create_session(s, character=char, player=player)
        chat_service.close_session(s, sess)
        s.refresh(sess)
        assert sess.status == ChatSessionStatus.ABANDONED


# --- handle_player_message path ------------------------------------------


def test_handle_player_message_persists_both_turns():
    with SessionLocal() as s:
        char = _mk_character(s)
        player = _mk_player(s, "turn_p")
        sess = chat_service.get_or_create_session(s, character=char, player=player)

        with patch(
            "app.characters.chat_service.run_soul",
            return_value=_soul_says("hello visitor"),
        ), patch("app.characters.chat_service.run_subconscious", return_value=[]):
            result = chat_service.handle_player_message(
                s,
                chat_session=sess,
                character=char,
                player=player,
                text="hello",
            )

        assert result.player_turn.text == "hello"
        assert result.character_turn.text == "hello visitor"
        assert result.character_turn.role == ChatTurnRole.CHARACTER
        turns = chat_service.get_turns(s, sess)
        assert [t.role for t in turns] == [ChatTurnRole.PLAYER, ChatTurnRole.CHARACTER]
        assert [t.turn_number for t in turns] == [1, 2]


def test_handle_player_message_empty_text_raises():
    with SessionLocal() as s:
        char = _mk_character(s)
        player = _mk_player(s, "emp_p")
        sess = chat_service.get_or_create_session(s, character=char, player=player)
        with pytest.raises(ValueError):
            chat_service.handle_player_message(
                s, chat_session=sess, character=char, player=player, text=""
            )


def test_handle_player_message_renders_silent_soul_as_ellipsis():
    with SessionLocal() as s:
        char = _mk_character(s)
        player = _mk_player(s, "sil_p")
        sess = chat_service.get_or_create_session(s, character=char, player=player)
        with patch(
            "app.characters.chat_service.run_soul",
            return_value=_soul_says(None),
        ), patch("app.characters.chat_service.run_subconscious", return_value=[]):
            result = chat_service.handle_player_message(
                s, chat_session=sess, character=char, player=player, text="hi",
            )
        assert result.character_turn.text == "…"


def test_handle_player_message_stores_emotion():
    with SessionLocal() as s:
        char = _mk_character(s)
        player = _mk_player(s, "emo_p")
        sess = chat_service.get_or_create_session(s, character=char, player=player)
        with patch(
            "app.characters.chat_service.run_soul",
            return_value=_soul_says("scoffs", emotion="smug", intensity=0.8),
        ), patch("app.characters.chat_service.run_subconscious", return_value=[]):
            result = chat_service.handle_player_message(
                s, chat_session=sess, character=char, player=player, text="hi",
            )
        assert result.character_turn.emotion == "smug"
        assert result.character_turn.emotion_intensity == 0.8


# --- hand-off to a real match --------------------------------------------


def test_start_game_creates_match_and_hands_off():
    with SessionLocal() as s:
        char = _mk_character(s)
        player = _mk_player(s, "start_p")
        sess = chat_service.get_or_create_session(s, character=char, player=player)

        soul_response = _soul_says(
            "alright, sit down", emotion="focused", intensity=0.6,
            game_action="start_game",
        )
        with patch(
            "app.characters.chat_service.run_soul",
            return_value=soul_response,
        ), patch("app.characters.chat_service.run_subconscious", return_value=[]):
            result = chat_service.handle_player_message(
                s, chat_session=sess, character=char, player=player, text="yeah let's play",
            )

        assert result.handed_off_match_id is not None

        fresh_sess = s.get(CharacterChatSession, sess.id)
        assert fresh_sess.status == ChatSessionStatus.HANDED_OFF
        assert fresh_sess.handed_off_match_id == result.handed_off_match_id


def test_propose_game_does_not_create_match():
    with SessionLocal() as s:
        char = _mk_character(s)
        player = _mk_player(s, "prop_p")
        sess = chat_service.get_or_create_session(s, character=char, player=player)
        with patch(
            "app.characters.chat_service.run_soul",
            return_value=_soul_says(
                "shall we play?", game_action="propose_game",
            ),
        ), patch("app.characters.chat_service.run_subconscious", return_value=[]):
            result = chat_service.handle_player_message(
                s, chat_session=sess, character=char, player=player, text="hello",
            )
        assert result.handed_off_match_id is None
        fresh_sess = s.get(CharacterChatSession, sess.id)
        assert fresh_sess.status == ChatSessionStatus.ACTIVE


def test_opponent_notes_forwarded_into_match_extra_state():
    from app.models.match import Match

    with SessionLocal() as s:
        char = _mk_character(s)
        player = _mk_player(s, "notes_p")
        sess = chat_service.get_or_create_session(s, character=char, player=player)

        soul_1 = SoulResponse(
            speak="interesting",
            emotion="focused",
            emotion_intensity=0.4,
            note_about_opponent="opens with chitchat; seems nervous",
        )
        with patch("app.characters.chat_service.run_soul", return_value=soul_1), \
             patch("app.characters.chat_service.run_subconscious", return_value=[]):
            chat_service.handle_player_message(
                s, chat_session=sess, character=char, player=player, text="nice to meet you",
            )

        soul_2 = SoulResponse(
            speak="let's play",
            emotion="focused",
            emotion_intensity=0.6,
            game_action="start_game",
        )
        with patch("app.characters.chat_service.run_soul", return_value=soul_2), \
             patch("app.characters.chat_service.run_subconscious", return_value=[]):
            r = chat_service.handle_player_message(
                s, chat_session=sess, character=char, player=player, text="sure",
            )

        match = s.get(Match, r.handed_off_match_id)
        notes = (match.extra_state or {}).get("pending_opponent_notes") or []
        assert "opens with chitchat; seems nervous" in notes
        assert match.extra_state.get("pre_match_chat_session_id") == sess.id


def test_chat_route_hand_off_returns_202_and_creates_match(monkeypatch):
    """Block 2: HTTP chat POST now returns 202 Accepted.

    The opening move is no longer run synchronously here — it fires on
    /play socket connect. Asserts: status 202, match created with
    move_count==0 (engine move fires later on socket connect).
    """
    import app.matches.service as match_service_mod
    from app.engine.registry import reset_engines_for_testing
    from app.main import create_app
    from app.models.match import Color, Match
    from app.redis_client import reset_memory_store_for_testing
    from fastapi.testclient import TestClient
    from tests.conftest import signup_and_login

    reset_engines_for_testing()
    reset_memory_store_for_testing()

    # Force character to white by pinning the "random" side to WHITE for
    # the player → so character plays black — wait, invert: player=BLACK
    # means character=WHITE. Pin random.choice to Color.BLACK.
    monkeypatch.setattr(
        match_service_mod.random, "choice", lambda _opts: Color.BLACK
    )
    # Force MockEngine so the test doesn't need Maia/Stockfish binaries.
    monkeypatch.setattr(
        "app.matches.service.available_engines", lambda: ["mock"]
    )

    # Stub out the post-move agent pipeline (Soul/Subconscious/emotion).
    async def _noop_async(*a, **kw):
        return None

    def _sync_noop(*a, **kw):
        return None

    monkeypatch.setattr(match_service_mod, "_run_agents_sync", _sync_noop)

    client = TestClient(create_app(), follow_redirects=False)
    signup_and_login(client, "handoff_user")

    with SessionLocal() as s:
        char = _mk_character(s)
        char_id = char.id

    soul_resp = _soul_says(
        "sit down", emotion="focused", intensity=0.6, game_action="start_game",
    )
    with patch(
        "app.characters.chat_service.run_soul", return_value=soul_resp,
    ), patch(
        "app.characters.chat_service.run_subconscious", return_value=[],
    ):
        r = client.post(
            f"/characters/{char_id}/chat", data={"text": "let's go"},
        )

    assert r.status_code == 202, r.text
    match_id = r.json()["handed_off_match_id"]
    assert match_id is not None

    # Opening move is no longer run synchronously on the HTTP path.
    # It fires as a background task when the client connects to /play.
    with SessionLocal() as s:
        m = s.get(Match, match_id)
        assert m is not None
        assert m.player_color == Color.BLACK  # character = WHITE
        assert m.move_count == 0, (
            "opening move should not be committed synchronously; "
            "it fires on /play socket connect (Block 2 change)"
        )


def test_pre_match_chat_seeds_table_talk_on_match_page(monkeypatch):
    """Regression (demo-rescue Fix 3): after a chat hand-off, the match
    page must ship the pre-match transcript to the browser so play.html
    can seed the chat log. The conversation should not visually reset."""
    import app.matches.service as match_service_mod
    from app.engine.registry import reset_engines_for_testing
    from app.main import create_app
    from app.models.match import Color
    from app.redis_client import reset_memory_store_for_testing
    from fastapi.testclient import TestClient
    from tests.conftest import signup_and_login

    reset_engines_for_testing()
    reset_memory_store_for_testing()
    # Pin player to WHITE so character = BLACK → no opening engine move
    # to worry about; this test focuses on the chat transcript only.
    monkeypatch.setattr(
        match_service_mod.random, "choice", lambda _opts: Color.WHITE
    )
    monkeypatch.setattr(
        "app.matches.service.available_engines", lambda: ["mock"]
    )

    async def _noop_async(*a, **kw):
        return None

    def _sync_noop(*a, **kw):
        return None

    monkeypatch.setattr(match_service_mod, "_run_agents_sync", _sync_noop)

    client = TestClient(create_app(), follow_redirects=False)
    signup_and_login(client, "premchat_user")

    with SessionLocal() as s:
        char = _mk_character(s)
        char_id = char.id

    # Turn 1: player chats, character greets (no game yet).
    with patch(
        "app.characters.chat_service.run_soul",
        return_value=_soul_says("nice to see you"),
    ), patch(
        "app.characters.chat_service.run_subconscious", return_value=[],
    ):
        r = client.post(
            f"/characters/{char_id}/chat", data={"text": "hey there"},
        )
    assert r.status_code == 202, r.text

    # Turn 2: player asks to play, Soul emits start_game → hand-off.
    with patch(
        "app.characters.chat_service.run_soul",
        return_value=_soul_says(
            "let's go", game_action="start_game",
        ),
    ), patch(
        "app.characters.chat_service.run_subconscious", return_value=[],
    ):
        r = client.post(
            f"/characters/{char_id}/chat", data={"text": "let's play"},
        )
    assert r.status_code == 202, r.text
    match_id = r.json()["handed_off_match_id"]
    assert match_id is not None

    # The board page must ship the transcript to the browser so play.html
    # can seed the chat log from PRE_MATCH_CHAT.
    r = client.get(f"/matches/{match_id}")
    assert r.status_code == 200, r.text
    assert "const PRE_MATCH_CHAT" in r.text
    assert "PRE_MATCH_CHAT = []" not in r.text
    # Both player and character lines must be in the serialized JS array.
    # Jinja's | tojson unicode-escapes apostrophes as \u0027, so match the
    # apostrophe-free turns directly.
    assert "hey there" in r.text
    assert "nice to see you" in r.text
    # Apostrophe-escaped turn: "let's go" → "let\u0027s go" in JSON.
    assert "let" in r.text and "s go" in r.text
