"""H.6: Two-user manual smoke test for Phase 3a.

Simulates two users (alice, bob) against a fresh tmp DB via FastAPI's
TestClient. Covers:
- login (creates accounts)
- each user creates a character
- alice's public character is visible to bob; alice's private is not
- bob can clone alice's character
- bob cannot edit or delete alice's character (403)
- alice can edit her own character
- alice can delete her own character
- neither can delete or edit a preset

Run with:
    DATABASE_URL=sqlite:///tmp_smoke.db python scripts/smoke_3a.py

Prints a pass/fail report to stdout and exits 0 on success.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="smoke3a_"))
    db_path = tmp / "smoke.db"
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path.as_posix()}"
    os.environ.setdefault("GEMINI_API_KEY", "smoke-key-not-real")
    os.environ.setdefault("LOG_DIR", str(tmp / "logs"))

    from fastapi.testclient import TestClient

    from app.db import SessionLocal, init_db
    from app.main import create_app
    from app.models.character import Character, CharacterState, ContentRating, Visibility

    init_db()

    # Seed one preset-like character so we can exercise preset protection.
    with SessionLocal() as s:
        preset = Character(
            name="Viktor the Preset",
            short_description="smoke preset",
            backstory="smoke",
            avatar_emoji="♜",
            target_elo=2000,
            current_elo=2000,
            floor_elo=2000,
            max_elo=2200,
            is_preset=True,
            preset_key="smoke_viktor",
            content_rating=ContentRating.MATURE,
            state=CharacterState.READY,
        )
        s.add(preset)
        s.commit()
        s.refresh(preset)
        preset_id = preset.id

    app = create_app()
    passed: list[str] = []
    failed: list[str] = []

    def ok(name: str, cond: bool, detail: str = "") -> None:
        if cond:
            passed.append(name)
            print(f"  PASS  {name}")
        else:
            failed.append(f"{name}: {detail}")
            print(f"  FAIL  {name}  {detail}")

    def login(username: str, *, rating: ContentRating = ContentRating.MATURE) -> TestClient:
        c = TestClient(app, follow_redirects=False)
        r = c.post("/login", data={"username": username, "next": "/"})
        assert r.status_code == 303, f"login failed: {r.status_code} {r.text}"
        # Bump rating so mature characters are visible.
        c.post("/settings", data={"display_name": username, "max_content_rating": rating.value})
        return c

    print("\n== User 1 logs in (alice) ==")
    alice = login("alice")
    alice_me = alice.get("/api/me").json()
    ok("alice /api/me username", alice_me["username"] == "alice", str(alice_me))
    ok("alice display_name defaults to username", alice_me["display_name"] == "alice")

    print("\n== User 2 logs in (bob) ==")
    bob = login("bob")
    bob_me = bob.get("/api/me").json()
    ok("bob /api/me username", bob_me["username"] == "bob")

    print("\n== Alice creates a public character ==")
    # Skip create via API (which kicks off LLM generation) — go direct via DB
    # so we don't need a live LLM for the smoke test.
    with SessionLocal() as s:
        alice_pub = Character(
            name="Alice Public",
            short_description="",
            backstory="x",
            target_elo=1400,
            current_elo=1400,
            floor_elo=1400,
            max_elo=1800,
            owner_id=alice_me["id"],
            visibility=Visibility.PUBLIC,
            content_rating=ContentRating.FAMILY,
            state=CharacterState.READY,
        )
        alice_priv = Character(
            name="Alice Private",
            short_description="",
            backstory="x",
            target_elo=1400,
            current_elo=1400,
            floor_elo=1400,
            max_elo=1800,
            owner_id=alice_me["id"],
            visibility=Visibility.PRIVATE,
            content_rating=ContentRating.FAMILY,
            state=CharacterState.READY,
        )
        s.add_all([alice_pub, alice_priv])
        s.commit()
        s.refresh(alice_pub)
        s.refresh(alice_priv)
        alice_pub_id = alice_pub.id
        alice_priv_id = alice_priv.id

    print("\n== Bob creates his own character ==")
    with SessionLocal() as s:
        bob_char = Character(
            name="Bob Public",
            short_description="",
            backstory="x",
            target_elo=1400,
            current_elo=1400,
            floor_elo=1400,
            max_elo=1800,
            owner_id=bob_me["id"],
            visibility=Visibility.PUBLIC,
            content_rating=ContentRating.FAMILY,
            state=CharacterState.READY,
        )
        s.add(bob_char)
        s.commit()
        s.refresh(bob_char)
        bob_char_id = bob_char.id

    print("\n== Visibility: bob sees alice's public; bob does NOT see alice's private ==")
    bob_listing = bob.get("/api/characters").json()
    bob_ids = {c["id"] for c in bob_listing}
    ok("bob sees alice_pub", alice_pub_id in bob_ids)
    ok("bob does NOT see alice_priv", alice_priv_id not in bob_ids)

    r = bob.get(f"/api/characters/{alice_priv_id}")
    ok("bob gets 404 on alice_priv detail", r.status_code == 404)

    print("\n== Clone: bob clones alice's public character ==")
    r = bob.post(f"/api/characters/{alice_pub_id}/clone")
    ok("clone returns 202", r.status_code == 202, r.text)
    if r.status_code == 202:
        clone = r.json()
        ok("clone owner is bob", clone["owner_id"] == bob_me["id"])
        ok("clone id differs from source", clone["id"] != alice_pub_id)

    print("\n== Authorization: bob cannot edit or delete alice's character ==")
    r = bob.patch(f"/api/characters/{alice_pub_id}", json={"short_description": "hacked"})
    ok("bob PATCH alice_pub -> 403", r.status_code == 403)
    r = bob.delete(f"/api/characters/{alice_pub_id}")
    ok("bob DELETE alice_pub -> 403", r.status_code == 403)

    print("\n== Alice can edit and delete her own character ==")
    r = alice.patch(f"/api/characters/{alice_pub_id}", json={"short_description": "updated"})
    ok("alice PATCH alice_pub -> 200", r.status_code == 200)
    if r.status_code == 200:
        ok("patched desc applied", r.json()["short_description"] == "updated")
    r = alice.delete(f"/api/characters/{alice_pub_id}")
    ok("alice DELETE alice_pub -> 204", r.status_code == 204)

    print("\n== Preset protection: neither user can edit or delete a preset ==")
    r = alice.patch(f"/api/characters/{preset_id}", json={"short_description": "x"})
    ok("alice PATCH preset -> 403", r.status_code == 403)
    r = bob.delete(f"/api/characters/{preset_id}")
    ok("bob DELETE preset -> 403", r.status_code == 403)

    print("\n== Summary ==")
    print(f"passed: {len(passed)}   failed: {len(failed)}")
    if failed:
        print("Failures:")
        for f in failed:
            print(f"  - {f}")
        return 1
    print("ALL GREEN")
    return 0


if __name__ == "__main__":
    sys.exit(main())
