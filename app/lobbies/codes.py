"""Short, human-friendly invite codes for lobbies (Phase 4.2d).

Design:
- 6 characters, uppercase
- Alphabet excludes visually ambiguous glyphs: 0/O, 1/I/L
- Usable alphabet is 31 chars → 31^6 ≈ 887M codes → collisions are rare
  enough that a simple "try a few random codes" loop is safe.

Kept here (not in `service.py`) so tests can exercise the shape + collision
handling in isolation.
"""

from __future__ import annotations

import secrets
from typing import Callable

# Drop ambiguous glyphs: 0/O, 1/I/L. "Z" kept — hard to confuse in uppercase.
_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"  # 31 chars
_DEFAULT_LENGTH = 6


def generate_code(length: int = _DEFAULT_LENGTH) -> str:
    """Generate a single random code. Uniqueness is the caller's
    responsibility (see `generate_unique_code`)."""
    return "".join(secrets.choice(_ALPHABET) for _ in range(length))


def generate_unique_code(
    exists: Callable[[str], bool],
    *,
    max_attempts: int = 12,
    length: int = _DEFAULT_LENGTH,
) -> str:
    """Generate codes until `exists(code)` returns False.

    Parameters
    ----------
    exists:
        Callable taking a code and returning True iff another lobby already
        has that code. The service passes a function that checks the DB.
    max_attempts:
        Collision is astronomically unlikely at a reasonable code length,
        so this should never be reached in practice — but bail rather than
        loop forever if something's badly wrong (e.g. alphabet size
        misconfigured).

    Raises
    ------
    RuntimeError
        If no unique code was found within max_attempts.
    """
    for _ in range(max_attempts):
        candidate = generate_code(length)
        if not exists(candidate):
            return candidate
    raise RuntimeError(
        f"Could not generate a unique {length}-char lobby code after "
        f"{max_attempts} attempts — investigate the alphabet / exists() function."
    )
