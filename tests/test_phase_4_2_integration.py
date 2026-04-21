"""Phase 4.2h — end-to-end HTTP integration tests for lobby + PvP flow."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.db import SessionLocal
from app.lobbies.service import active_members
from app.main import create_app
from app.models.lobby import Lobby, LobbyStatus
from app.models.match import Player
from tests.conftest import signup_and_login


def _client() -> TestClient:
    return TestClient(create_app(), follow_redirects=False)


# --- create → join → controls → start → play → resign ---------------------


def test_full_lobby_and_pvp_flow_autostart():
    """Phase 4.2.5 — the Start button is gone. When both seats fill,
    the lobby GET auto-starts a match and redirects to the board.
    """
    host_c = _client()
    guest_c = _client()
    signup_and_login(host_c, "flow_host")
    signup_and_login(guest_c, "flow_guest")

    # Host creates a public room.
    r = host_c.post(
        "/lobby/new",
        data={
            "is_private": "",
            "allow_spectators": "on",
            "music_track": "cafe_murmur",
            "music_volume": 0.4,
            "lights_brightness": 0.6,
            "lights_hue": "#A0FF20",
        },
    )
    lobby_id = r.headers["location"].rsplit("/", 1)[-1]

    # Host toggles door to private mid-flight — should persist.
    r = host_c.post(f"/lobby/{lobby_id}/controls", data={"is_private": "true"})
    assert r.status_code == 200

    # Guest attempting to change controls is forbidden.
    r = guest_c.post(f"/lobby/{lobby_id}/controls", data={"lights_hue": "#FF00FF"})
    assert r.status_code == 403

    # Private now, so guest must enter via code.
    with SessionLocal() as s:
        code = s.get(Lobby, lobby_id).code
    r = guest_c.post("/lobby/join", data={"code": code})
    assert r.status_code == 303
    assert r.headers["location"] == f"/lobby/{lobby_id}"

    # Host visits the lobby page — both seats filled → auto-start → redirect.
    r = host_c.get(f"/lobby/{lobby_id}")
    assert r.status_code == 303
    assert r.headers["location"].startswith("/pvp/")
    match_id = r.headers["location"].rsplit("/", 1)[-1]

    # Guest visiting after auto-start is bounced into the same PvP page.
    r = guest_c.get(f"/lobby/{lobby_id}")
    assert r.status_code == 303
    assert r.headers["location"] == f"/pvp/{match_id}"

    # Either player plays first. Identify who's white.
    with SessionLocal() as s:
        from app.models.lobby import PvpMatch

        match = s.get(PvpMatch, match_id)
        white_is_host = match.white_player_id == (
            s.query(Player).filter_by(username="flow_host").one().id
        )
    mover_c = host_c if white_is_host else guest_c
    other_c = guest_c if white_is_host else host_c

    r = mover_c.post(f"/pvp/{match_id}/move", data={"uci": "e2e4"})
    assert r.status_code == 200
    assert r.json()["move"]["san"] == "e4"

    # Wrong side move rejected.
    r = mover_c.post(f"/pvp/{match_id}/move", data={"uci": "d2d4"})
    assert r.status_code == 400
    assert r.json()["error"] == "pvp_not_your_turn"

    # Other player plays e5.
    r = other_c.post(f"/pvp/{match_id}/move", data={"uci": "e7e5"})
    assert r.status_code == 200

    # Resign from the white side.
    r = mover_c.post(f"/pvp/{match_id}/resign")
    assert r.status_code == 200

    with SessionLocal() as s:
        # Private match → Elo untouched.
        host = s.query(Player).filter_by(username="flow_host").one()
        guest = s.query(Player).filter_by(username="flow_guest").one()
        assert host.elo == 1200
        assert guest.elo == 1200
        # Lobby flipped back to OPEN for a rematch.
        lob = s.get(Lobby, lobby_id)
        assert lob.status == LobbyStatus.OPEN


def test_private_lobby_join_requires_code():
    host_c = _client()
    guest_c = _client()
    signup_and_login(host_c, "pv_host")
    signup_and_login(guest_c, "pv_guest")

    r = host_c.post(
        "/lobby/new",
        data={
            "is_private": "on",
            "allow_spectators": "on",
            "music_volume": 0.4,
            "lights_brightness": 0.6,
            "lights_hue": "#C9A66B",
        },
    )
    lobby_id = r.headers["location"].rsplit("/", 1)[-1]
    with SessionLocal() as s:
        code = s.get(Lobby, lobby_id).code

    # Guest hitting /lobby/{id} directly is bounced to /lobby/join.
    r = guest_c.get(f"/lobby/{lobby_id}")
    assert r.status_code == 303
    assert "/lobby/join" in r.headers["location"]

    # Correct code gets them in.
    r = guest_c.post("/lobby/join", data={"code": code})
    assert r.status_code == 303
    assert r.headers["location"] == f"/lobby/{lobby_id}"


def test_matchmaking_pairs_via_http():
    a_c = _client()
    b_c = _client()
    signup_and_login(a_c, "mm_http_a")
    signup_and_login(b_c, "mm_http_b")

    # Both hit matchmaking/enqueue — the second hit should pair them.
    a_c.post("/matchmaking/enqueue")
    b_c.post("/matchmaking/enqueue")

    r = a_c.get("/matchmaking/poll")
    assert r.status_code == 200
    body = r.json()
    assert body["matched_lobby_id"] is not None

    r2 = b_c.get("/matchmaking/poll")
    assert r2.json()["matched_lobby_id"] == body["matched_lobby_id"]


def test_public_lobbies_index_shows_open_rooms():
    host_c = _client()
    signup_and_login(host_c, "list_host")
    host_c.post(
        "/lobby/new",
        data={
            "allow_spectators": "on",
            "music_volume": 0.5,
            "lights_brightness": 0.7,
            "lights_hue": "#C9A66B",
        },
    )

    viewer_c = _client()
    signup_and_login(viewer_c, "list_view")
    r = viewer_c.get("/lobbies")
    assert r.status_code == 200
    assert "Open lobbies" in r.text
    assert "@list_host" in r.text


def test_host_leave_transfers_host_over_http():
    """Host-transfer is still a required property, but auto-start now
    happens the instant guest GETs the page. We avoid the GET by using
    the lobby-service directly to seat the guest, then exercise the
    /leave route on a still-open lobby."""
    from app.lobbies.service import (
        CreateLobbyIn,
        create_lobby,
        join_lobby,
    )

    host_c = _client()
    signup_and_login(host_c, "t_host")

    with SessionLocal() as s:
        host = s.query(Player).filter_by(username="t_host").one()
        guest = Player(username="t_guest", display_name="Guest", elo=1200)
        s.add(guest)
        s.commit()
        lob = create_lobby(s, host, CreateLobbyIn())
        join_lobby(s, lob, guest)
        lobby_id = lob.id
        guest_id = guest.id

    # Host leaves without visiting the page (no auto-start trigger).
    r = host_c.post(f"/lobby/{lobby_id}/leave")
    assert r.status_code == 303

    with SessionLocal() as s:
        lob = s.get(Lobby, lobby_id)
        assert lob.host_id == guest_id
        assert lob.status == LobbyStatus.OPEN
