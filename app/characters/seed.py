"""Idempotent preset seeding.

Called on app startup. Creates any missing preset characters and kicks
off memory generation for any whose state isn't `ready`.
"""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime

from sqlalchemy import select

from app.characters.presets import PRESETS, PresetSpec
from app.db import session_scope
from app.models.character import Character, CharacterState

# Fixed namespace for stable preset UUIDs — never change this.
_PRESET_NS = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")


def _preset_id(preset_key: str) -> str:
    """Deterministic UUID derived from preset_key — survives DB resets."""
    return str(uuid.uuid5(_PRESET_NS, preset_key))

logger = logging.getLogger(__name__)


def _upsert_preset(spec: PresetSpec) -> tuple[str, bool]:
    """Insert the preset if missing. Returns (character_id, created)."""
    with session_scope() as session:
        existing = session.execute(
            select(Character).where(Character.preset_key == spec.preset_key)
        ).scalar_one_or_none()
        if existing is not None:
            return existing.id, False

        from app.models.character import Visibility

        char = Character(
            id=_preset_id(spec.preset_key),
            name=spec.name,
            short_description=spec.short_description,
            backstory=spec.backstory,
            avatar_emoji=spec.avatar_emoji,
            aggression=spec.aggression,
            risk_tolerance=spec.risk_tolerance,
            patience=spec.patience,
            trash_talk=spec.trash_talk,
            target_elo=spec.target_elo,
            current_elo=spec.target_elo,
            floor_elo=spec.target_elo,
            max_elo=spec.max_elo,
            adaptive=spec.adaptive,
            opening_preferences=list(spec.opening_preferences),
            voice_descriptor=spec.voice_descriptor,
            quirks=spec.quirks,
            is_preset=True,
            preset_key=spec.preset_key,
            owner_id=None,  # presets are system-owned
            visibility=Visibility.PUBLIC,
            content_rating=spec.content_rating,
            state=CharacterState.GENERATING_MEMORIES,
            memory_generation_started_at=datetime.utcnow(),
        )
        session.add(char)
        session.flush()
        return char.id, True


def _needs_generation(character_id: str) -> bool:
    with session_scope() as session:
        char = session.get(Character, character_id)
        if char is None:
            return False
        return char.state != CharacterState.READY


def _run_generation(character_id: str) -> None:
    # Imported lazily so tests that don't need the LLM wrapper never touch google-genai.
    from app.characters.memory_generator import generate_and_store

    try:
        generate_and_store(character_id)
    except Exception:
        # generate_and_store already marks the character as generation_failed.
        logger.exception("Background memory generation failed for %s", character_id)


def seed_presets(*, run_generation: bool = True) -> dict[str, bool]:
    """Seed preset characters. Returns {preset_key: created_now?}.

    Idempotent: existing presets are left alone. Missing or failed-generation
    presets have their memory generation kicked off in a background thread
    when `run_generation` is True.
    """
    results: dict[str, bool] = {}
    for spec in PRESETS:
        character_id, created = _upsert_preset(spec)
        results[spec.preset_key] = created

        if run_generation and _needs_generation(character_id):
            thread = threading.Thread(
                target=_run_generation,
                args=(character_id,),
                name=f"seed-gen-{spec.preset_key}",
                daemon=True,
            )
            thread.start()

    return results
