"""Cookie-based player auth helpers (Phase 3a).

The cookie contains a player UUID. There's no password — usernames are
claimed on first login. `require_player` raises `NotAuthenticated` which
the app's exception handler translates to:
- 302 redirect to /login for HTML routes
- 401 JSON for /api/* routes
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

from fastapi import Cookie, Depends
from sqlalchemy.orm import Session

from app.db import get_session
from app.models.match import (
    GUEST_USERNAME_PREFIX,
    USERNAME_MAX_LEN,
    USERNAME_MIN_LEN,
    USERNAME_PATTERN,
    Player,
)

PLAYER_COOKIE = "player_id"
COOKIE_MAX_AGE = 60 * 60 * 24 * 365  # one year

_USERNAME_RE = re.compile(USERNAME_PATTERN)


class NotAuthenticated(Exception):
    """Raised when a protected route is hit without a valid player cookie."""


@dataclass
class UsernameError(ValueError):
    code: str
    message: str

    def __str__(self) -> str:
        return self.message


def normalize_username(raw: str) -> str:
    """Lowercase + strip. Does not validate — use `validate_username` for that."""
    return (raw or "").strip().lower()


def validate_username(username: str) -> None:
    """Raises UsernameError if the username isn't well-formed."""
    if not username:
        raise UsernameError(code="empty", message="Username cannot be blank.")
    if len(username) < USERNAME_MIN_LEN:
        raise UsernameError(
            code="too_short",
            message=f"Username must be at least {USERNAME_MIN_LEN} characters.",
        )
    if len(username) > USERNAME_MAX_LEN:
        raise UsernameError(
            code="too_long",
            message=f"Username must be at most {USERNAME_MAX_LEN} characters.",
        )
    if not _USERNAME_RE.match(username):
        raise UsernameError(
            code="invalid_chars",
            message="Usernames may only contain lowercase letters, digits, and underscores.",
        )


def is_guest_username(username: str) -> bool:
    return username.startswith(GUEST_USERNAME_PREFIX)


def generate_guest_username() -> str:
    return f"{GUEST_USERNAME_PREFIX}{uuid.uuid4().hex[:8]}"


def get_optional_player(
    player_id: str | None = Cookie(default=None, alias=PLAYER_COOKIE),
    session: Session = Depends(get_session),
) -> Player | None:
    """Dependency for routes that render differently for logged-in vs logged-out."""
    if not player_id:
        return None
    return session.get(Player, player_id)


def require_player(
    player_id: str | None = Cookie(default=None, alias=PLAYER_COOKIE),
    session: Session = Depends(get_session),
) -> Player:
    """Dependency that guarantees a logged-in player or raises NotAuthenticated."""
    if not player_id:
        raise NotAuthenticated()
    player = session.get(Player, player_id)
    if player is None:
        raise NotAuthenticated()
    return player


def find_player_by_username(session: Session, username: str) -> Player | None:
    """Case-insensitive exact lookup. Returns the Player or None."""
    from sqlalchemy import select

    return session.execute(
        select(Player).where(Player.username == username)
    ).scalar_one_or_none()
