"""End-to-end live verification.

Run against a running `docker compose up` stack. Exercises the full
player flow against Archibald (a Maia-range character) and asserts
the engine response comes from Maia-2 or Stockfish, not MockEngine.

Usage:
    python scripts/verify_live_play.py [base_url]

Default base URL: http://localhost:8000
"""

from __future__ import annotations

import sys
import time
from typing import Any
from urllib.parse import urljoin

import httpx


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main(base_url: str) -> None:
    client = httpx.Client(base_url=base_url, timeout=30.0, follow_redirects=False)

    # 0. Wait for the app to be reachable
    for attempt in range(20):
        try:
            r = client.get("/api/characters")
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(1.5)
    else:
        fail(f"App at {base_url} never returned 200 on /api/characters")

    chars: list[dict[str, Any]] = r.json()
    print(f"  characters loaded: {len(chars)}")
    archibald = next((c for c in chars if "Archibald" in c["name"]), None)
    if archibald is None:
        fail("Archibald preset missing — expected 4 presets seeded at startup")

    # 1. Create a match as white
    r = client.post("/api/matches", json={"character_id": archibald["id"], "player_color": "white"})
    if r.status_code != 201:
        fail(f"POST /api/matches returned {r.status_code}: {r.text}")
    match = r.json()
    match_id = match["id"]
    print(f"  match created: {match_id}, you=white vs {archibald['name']}")

    # 2. Submit e2e4
    r = client.post(f"/api/matches/{match_id}/move", json={"uci": "e2e4"})
    if r.status_code != 200:
        fail(f"POST /move returned {r.status_code}: {r.text}")
    body = r.json()
    agent_move = body["agent_move"]
    if not agent_move:
        fail("No agent move returned — game ended on player's move?")

    engine_name = agent_move["engine_name"]
    san = agent_move["san"]
    ms = agent_move["time_taken_ms"]
    print(f"  agent replied: {san} ({engine_name}, {ms}ms)")

    if engine_name == "mock":
        fail(
            "Agent moved with MockEngine — real engine (Maia-2 or Stockfish) "
            "failed to serve the move. Check /matches/{id} page for the banner."
        )

    if engine_name not in ("maia2", "stockfish"):
        fail(f"Unexpected engine: {engine_name}")

    # 3. Static asset check
    r = client.get("/static/img/chesspieces/wikipedia/wK.png")
    if r.status_code != 200 or len(r.content) < 500:
        fail(f"/static/img/chesspieces/wikipedia/wK.png: {r.status_code}, {len(r.content)}B")
    print(f"  piece PNG served: {len(r.content)}B, content-type={r.headers.get('content-type')}")

    # 4. Play page renders
    r = client.get(f"/matches/{match_id}")
    if r.status_code != 200:
        fail(f"/matches/{match_id} returned {r.status_code}")
    if "alert-banner" not in r.text or "HAS_REAL_ENGINE" not in r.text:
        fail("Play page missing expected template elements")
    print(f"  play page renders: {len(r.text)}B")

    print()
    print(f"LIVE PLAY OK — agent moved {san} via {engine_name}")


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    main(url)
