"""Cookie-based player auth helpers.

Phase 3a: username-only login, cookie stores player UUID, no passwords.
Phase 4.0a: adds email + password. `Player.password_hash` is nullable —
rows without one are in "legacy" mode and can still log in with just a
username (the UI nudges them to set a password). New signups require
email + username + password.

`require_player` raises `NotAuthenticated` which the app's exception
handler translates to:
- 303 redirect to /login for HTML routes
- 401 JSON for /api/* routes
"""

from __future__ import annotations

import hashlib
import re
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHash, VerifyMismatchError
from email_validator import EmailNotValidError, validate_email
from fastapi import Cookie, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_session
from app.models.auth import PasswordResetToken
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

# A single hasher instance — cheap to reuse. Argon2 defaults (time_cost=3,
# memory_cost=64MB, parallelism=4) are fine for a demo; tune later if login
# latency becomes a concern.
_PH = PasswordHasher()


class NotAuthenticated(Exception):
    """Raised when a protected route is hit without a valid player cookie."""


@dataclass
class UsernameError(ValueError):
    code: str
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass
class EmailError(ValueError):
    code: str
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass
class PasswordError(ValueError):
    code: str
    message: str

    def __str__(self) -> str:
        return self.message


# --- username --------------------------------------------------------------


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


# --- email -----------------------------------------------------------------


def normalize_email(raw: str) -> str:
    return (raw or "").strip().lower()


def validate_email_address(email: str) -> str:
    """Returns the normalized email or raises EmailError.

    Uses the `email-validator` package for RFC-ish correctness without
    DNS lookups (check_deliverability=False — we don't want tests to hit
    the network).
    """
    if not email:
        raise EmailError(code="empty", message="Email cannot be blank.")
    try:
        info = validate_email(email, check_deliverability=False)
    except EmailNotValidError as exc:
        raise EmailError(code="invalid", message=str(exc)) from exc
    return info.normalized.lower()


# --- password --------------------------------------------------------------


def validate_password(password: str) -> None:
    """Raises PasswordError if weak. Only enforces length for Phase 4.0a."""
    if not password:
        raise PasswordError(code="empty", message="Password cannot be blank.")
    min_len = get_settings().password_min_length
    if len(password) < min_len:
        raise PasswordError(
            code="too_short",
            message=f"Password must be at least {min_len} characters.",
        )
    if len(password) > 1024:
        # Guard against DoS via huge-string argon2 work. 1024 chars is absurdly
        # long but still way under the Argon2 spec limit.
        raise PasswordError(code="too_long", message="Password is too long.")


def hash_password(plaintext: str) -> str:
    return _PH.hash(plaintext)


def verify_password(plaintext: str, stored_hash: str | None) -> bool:
    """Constant-time-ish password verification. Returns False on any failure."""
    if not stored_hash:
        return False
    try:
        return _PH.verify(stored_hash, plaintext)
    except (VerifyMismatchError, InvalidHash):
        return False
    except Exception:
        # argon2-cffi raises broad exceptions on corrupted hashes — treat
        # any unexpected failure as "does not match" rather than 500ing.
        return False


def needs_rehash(stored_hash: str) -> bool:
    try:
        return _PH.check_needs_rehash(stored_hash)
    except Exception:
        return False


# --- identity cookie (unchanged from 3a) -----------------------------------


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
    return session.execute(
        select(Player).where(Player.username == username)
    ).scalar_one_or_none()


def find_player_by_email(session: Session, email: str) -> Player | None:
    return session.execute(
        select(Player).where(Player.email == email)
    ).scalar_one_or_none()


def find_player_by_identifier(session: Session, identifier: str) -> Player | None:
    """Accepts either a username or an email. Tries email first if it looks
    like one (contains '@'), otherwise username.
    """
    identifier = identifier.strip()
    if not identifier:
        return None
    if "@" in identifier:
        return find_player_by_email(session, identifier.lower())
    return find_player_by_username(session, identifier.lower())


# --- password reset tokens -------------------------------------------------


def _hash_token(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def issue_password_reset_token(session: Session, player: Player) -> str:
    """Creates a fresh reset token, persists its hash, returns the RAW token.

    The raw token must be emailed and then discarded — only the hash is
    persisted. Caller commits the session.
    """
    raw = secrets.token_urlsafe(32)
    ttl = get_settings().password_reset_token_ttl_minutes
    token = PasswordResetToken(
        player_id=player.id,
        token_hash=_hash_token(raw),
        expires_at=datetime.utcnow() + timedelta(minutes=ttl),
    )
    session.add(token)
    return raw


def consume_password_reset_token(
    session: Session, raw_token: str
) -> Player | None:
    """Look up a reset token, mark it used, return the owning Player.

    Returns None if the token is unknown, expired, or already consumed.
    Caller commits the session.
    """
    if not raw_token:
        return None
    row = session.execute(
        select(PasswordResetToken).where(
            PasswordResetToken.token_hash == _hash_token(raw_token)
        )
    ).scalar_one_or_none()
    if row is None:
        return None
    if row.used_at is not None:
        return None
    if row.expires_at < datetime.utcnow():
        return None
    row.used_at = datetime.utcnow()
    return session.get(Player, row.player_id)
