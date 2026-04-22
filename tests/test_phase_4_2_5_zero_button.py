"""Phase 4.2.5 — zero-button pass: auto-start PvP when both seated + clock control."""

from __future__ import annotations

import pytest

from app.db import SessionLocal
from app.lobbies.service import (
    ControlsPatch,
    CreateLobbyIn,
    LobbyInvalidControl,
    TIME_CONTROL_PRESETS,
    create_lobby,
    join_lobby,
    update_controls,
)
from app.models.lobby import Lobby
from app.models.match import Player


def _mk_player(s, username: str) -> Player:
    p = Player(username=username, display_name=username, elo=1200)
    s.add(p); s.commit(); s.refresh(p)
    return p


# --- tabletop clock control ----------------------------------------------


def test_clock_defaults_to_untimed():
    with SessionLocal() as s:
        host = _mk_player(s, "clk_1")
        lob = create_lobby(s, host, CreateLobbyIn())
        assert lob.time_control == "untimed"


def test_clock_cycle_accepts_known_presets():
    with SessionLocal() as s:
        host = _mk_player(s, "clk_2")
        lob = create_lobby(s, host, CreateLobbyIn())
        for preset in TIME_CONTROL_PRESETS:
            update_controls(s, lob, by=host, patch=ControlsPatch(time_control=preset))
            s.refresh(lob)
            assert lob.time_control == preset


def test_clock_rejects_unknown_preset():
    with SessionLocal() as s:
        host = _mk_player(s, "clk_3")
        lob = create_lobby(s, host, CreateLobbyIn())
        with pytest.raises(LobbyInvalidControl):
            update_controls(s, lob, by=host, patch=ControlsPatch(time_control="3+7"))


# --- auto-start on second seat -------------------------------------------


def test_lobby_state_endpoint_exposes_time_control(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import create_app
    from tests.conftest import signup_and_login

    c = TestClient(create_app(), follow_redirects=False)
    signup_and_login(c, "state_user")
    r = c.post("/lobby/new", data={
        "music_volume": 0.5, "lights_brightness": 0.7, "lights_hue": "#C9A66B",
    })
    lobby_id = r.headers["location"].rsplit("/", 1)[-1]
    r = c.get(f"/lobby/{lobby_id}/state")
    assert r.status_code == 200
    assert r.json()["time_control"] == "untimed"

    r = c.post(f"/lobby/{lobby_id}/controls", data={"time_control": "10+0"})
    assert r.status_code == 200
    assert r.json()["time_control"] == "10+0"


def test_lobby_ambient_script_unmutes_on_first_gesture():
    """Fix 4 (demo-rescue): lobby_ambient.js must implicitly unmute on
    the first user gesture when no explicit preference is stored. The
    coffee_shop.ogg bed otherwise plays at volume 0 and looks broken.
    Lock in the implementation so a future refactor doesn't silently
    revert to opt-in audio."""
    from pathlib import Path

    src = Path("app/web/static/js/lobby_ambient.js").read_text(encoding="utf-8")
    # The unmute-on-first-gesture branch must call setItem("mp_ambient_muted", "false").
    assert "MUTED_KEY" in src
    assert 'setItem(MUTED_KEY, "false")' in src
    assert "onFirstGesture" in src
