"""Phase 4.2b — lobby service layer unit tests."""

from __future__ import annotations

import pytest

from app.db import SessionLocal
from app.lobbies.codes import generate_code, generate_unique_code
from app.lobbies.service import (
    ControlsPatch,
    CreateLobbyIn,
    LobbyClosed,
    LobbyForbidden,
    LobbyFull,
    LobbyInvalidControl,
    LobbyNotFound,
    active_members,
    close_lobby,
    create_lobby,
    current_active_lobby_for,
    get_lobby_by_code,
    join_lobby,
    join_lobby_by_code,
    leave_lobby,
    list_public_open_lobbies,
    update_controls,
)
from app.models.lobby import LobbyRole, LobbyStatus
from app.models.match import Player


def _mk_player(s, username: str, elo: int = 1200) -> Player:
    p = Player(username=username, display_name=username, elo=elo)
    s.add(p)
    s.commit()
    s.refresh(p)
    return p


# --- invite codes ---------------------------------------------------------


def test_generate_code_shape():
    for _ in range(50):
        c = generate_code()
        assert len(c) == 6
        for ch in c:
            assert ch in "ABCDEFGHJKMNPQRSTUVWXYZ23456789"


def test_generate_unique_code_retries_on_collision():
    seen = {"ABCDEF"}  # pretend this code is taken

    def exists(c):
        return c in seen

    # Generate many — should eventually return something new.
    code = generate_unique_code(exists, max_attempts=20)
    assert code != "ABCDEF"


def test_generate_unique_code_gives_up_on_perma_collision():
    with pytest.raises(RuntimeError):
        generate_unique_code(lambda _: True, max_attempts=3)


# --- create + join --------------------------------------------------------


def test_create_lobby_seats_host_and_generates_code():
    with SessionLocal() as s:
        host = _mk_player(s, "host1")
        lob = create_lobby(s, host, CreateLobbyIn(is_private=True))
        assert lob.host_id == host.id
        assert lob.is_private is True
        assert lob.status == LobbyStatus.OPEN
        assert lob.code and len(lob.code) == 6
        members = active_members(s, lob.id)
        assert len(members) == 1
        assert members[0].role == LobbyRole.HOST


def test_join_lobby_seats_guest():
    with SessionLocal() as s:
        host = _mk_player(s, "h_a")
        guest = _mk_player(s, "g_a")
        lob = create_lobby(s, host, CreateLobbyIn())
        join_lobby(s, lob, guest)
        members = active_members(s, lob.id)
        assert {m.player_id for m in members} == {host.id, guest.id}
        guest_mem = next(m for m in members if m.player_id == guest.id)
        assert guest_mem.role == LobbyRole.GUEST


def test_join_lobby_is_idempotent():
    with SessionLocal() as s:
        host = _mk_player(s, "h_b")
        guest = _mk_player(s, "g_b")
        lob = create_lobby(s, host, CreateLobbyIn())
        join_lobby(s, lob, guest)
        join_lobby(s, lob, guest)  # second join is a no-op
        members = active_members(s, lob.id)
        assert len(members) == 2


def test_third_player_cannot_join_full_lobby():
    with SessionLocal() as s:
        host = _mk_player(s, "h_c")
        guest = _mk_player(s, "g_c")
        bystander = _mk_player(s, "b_c")
        lob = create_lobby(s, host, CreateLobbyIn())
        join_lobby(s, lob, guest)
        with pytest.raises(LobbyFull):
            join_lobby(s, lob, bystander)


def test_join_by_code_finds_correct_lobby():
    with SessionLocal() as s:
        host = _mk_player(s, "h_d")
        guest = _mk_player(s, "g_d")
        lob = create_lobby(s, host, CreateLobbyIn(is_private=True))
        returned_lobby, _ = join_lobby_by_code(s, lob.code.lower(), guest)
        assert returned_lobby.id == lob.id


def test_join_by_code_rejects_unknown_code():
    with SessionLocal() as s:
        p = _mk_player(s, "p_nope")
        with pytest.raises(LobbyNotFound):
            join_lobby_by_code(s, "QQQQQQ", p)


def test_joining_new_lobby_removes_you_from_old_one():
    """One-lobby-at-a-time rule."""
    with SessionLocal() as s:
        alice = _mk_player(s, "alice")
        bob = _mk_player(s, "bob")
        carol = _mk_player(s, "carol")

        lob1 = create_lobby(s, alice, CreateLobbyIn())
        join_lobby(s, lob1, bob)

        # Carol opens a second lobby; alice migrates in.
        lob2 = create_lobby(s, carol, CreateLobbyIn())
        join_lobby(s, lob2, alice)

        # Alice should no longer be in lob1. In lob1, bob is host now.
        active1 = active_members(s, lob1.id)
        assert alice.id not in [m.player_id for m in active1]
        # lob1 auto-transferred host to bob.
        s.refresh(lob1)
        assert lob1.host_id == bob.id


# --- leave + close --------------------------------------------------------


def test_host_leaving_transfers_host_to_remaining_member():
    with SessionLocal() as s:
        host = _mk_player(s, "h_l")
        guest = _mk_player(s, "g_l")
        lob = create_lobby(s, host, CreateLobbyIn())
        join_lobby(s, lob, guest)
        leave_lobby(s, lob, host)
        s.refresh(lob)
        assert lob.host_id == guest.id
        assert lob.status == LobbyStatus.OPEN


def test_last_member_leaving_closes_lobby():
    with SessionLocal() as s:
        host = _mk_player(s, "h_last")
        lob = create_lobby(s, host, CreateLobbyIn())
        leave_lobby(s, lob, host)
        s.refresh(lob)
        assert lob.status == LobbyStatus.CLOSED
        assert lob.closed_at is not None


def test_close_lobby_is_host_only():
    with SessionLocal() as s:
        host = _mk_player(s, "h_k")
        guest = _mk_player(s, "g_k")
        lob = create_lobby(s, host, CreateLobbyIn())
        join_lobby(s, lob, guest)
        with pytest.raises(LobbyForbidden):
            close_lobby(s, lob, by=guest)


# --- controls -------------------------------------------------------------


def test_update_controls_host_only():
    with SessionLocal() as s:
        host = _mk_player(s, "h_ctl")
        guest = _mk_player(s, "g_ctl")
        lob = create_lobby(s, host, CreateLobbyIn())
        join_lobby(s, lob, guest)
        with pytest.raises(LobbyForbidden):
            update_controls(s, lob, by=guest, patch=ControlsPatch(is_private=True))


def test_update_controls_applies_partial_patch():
    with SessionLocal() as s:
        host = _mk_player(s, "h_ctl2")
        lob = create_lobby(s, host, CreateLobbyIn(lights_brightness=0.7, lights_hue="#C9A66B"))
        update_controls(
            s, lob, by=host,
            patch=ControlsPatch(music_track="rain_window", lights_hue="#A0FF20"),
        )
        s.refresh(lob)
        assert lob.music_track == "rain_window"
        assert lob.lights_hue == "#A0FF20"
        assert lob.lights_brightness == 0.7  # untouched


def test_update_controls_rejects_invalid_hex():
    with SessionLocal() as s:
        host = _mk_player(s, "h_ctl3")
        lob = create_lobby(s, host, CreateLobbyIn())
        with pytest.raises(LobbyInvalidControl):
            update_controls(s, lob, by=host, patch=ControlsPatch(lights_hue="not-a-hex"))


def test_update_controls_clamps_out_of_range_volume():
    with SessionLocal() as s:
        host = _mk_player(s, "h_vol")
        lob = create_lobby(s, host, CreateLobbyIn())
        with pytest.raises(LobbyInvalidControl):
            update_controls(s, lob, by=host, patch=ControlsPatch(music_volume=1.5))


def test_close_lobby_rejects_operations():
    with SessionLocal() as s:
        host = _mk_player(s, "h_closed")
        lob = create_lobby(s, host, CreateLobbyIn())
        close_lobby(s, lob, by=host)
        with pytest.raises(LobbyClosed):
            update_controls(s, lob, by=host, patch=ControlsPatch(music_track="none"))


# --- lookups --------------------------------------------------------------


def test_current_active_lobby_for():
    with SessionLocal() as s:
        host = _mk_player(s, "h_cal")
        lob = create_lobby(s, host, CreateLobbyIn())
        assert current_active_lobby_for(s, host.id).id == lob.id
        leave_lobby(s, lob, host)
        assert current_active_lobby_for(s, host.id) is None


def test_list_public_open_lobbies_excludes_private_and_closed():
    with SessionLocal() as s:
        a = _mk_player(s, "pub_a")
        b = _mk_player(s, "priv_b")
        c = _mk_player(s, "closed_c")
        pub = create_lobby(s, a, CreateLobbyIn(is_private=False))
        priv = create_lobby(s, b, CreateLobbyIn(is_private=True))
        gone = create_lobby(s, c, CreateLobbyIn(is_private=False))
        close_lobby(s, gone, by=c)

        listing = list_public_open_lobbies(s)
        ids = {l.id for l in listing}
        assert pub.id in ids
        assert priv.id not in ids
        assert gone.id not in ids


def test_get_lobby_by_code_is_case_insensitive():
    with SessionLocal() as s:
        host = _mk_player(s, "h_cs")
        lob = create_lobby(s, host, CreateLobbyIn())
        assert get_lobby_by_code(s, lob.code.lower()).id == lob.id
        assert get_lobby_by_code(s, lob.code.upper()).id == lob.id
