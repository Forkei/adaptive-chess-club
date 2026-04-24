"""Socket.IO real-time transport for gameplay (Phase 3b).

Public surface:

- `sio` — the `AsyncServer`. Imported by `app.main` and mounted with the FastAPI app.
- `build_asgi_app(fastapi_app)` — wrap a FastAPI app so both Socket.IO and HTTP share one ASGI entry.
- `emit_post_match_status` — thread-safe helper for the post-match processor callback.
- `match_room` — deterministic room name for a given match id.

See `events.py` for the full client↔server event contract.
"""

from __future__ import annotations

from app.sockets.bridge import emit_post_match_status
from app.sockets.server import build_asgi_app, match_room, sio

# Importing for side-effects — registers /lobby namespace handlers on `sio`.
# The module itself is rarely imported directly; HTTP routes use its
# `broadcast_*` helpers through `from app.sockets.lobby_server import ...`.
from app.sockets import lobby_server  # noqa: F401

# Importing for side-effects — registers /room namespace handlers on `sio`.
from app.sockets import room_server  # noqa: F401

__all__ = ["sio", "build_asgi_app", "match_room", "emit_post_match_status"]
