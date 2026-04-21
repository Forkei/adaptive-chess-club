"""Deterministic mapping from style sliders to prompt fragments.

Phase 1 exposes this helper and exercises it in tests. Phase 2's Soul
prompt will assemble its system message by joining the dict values.

Sliders are integers 1-10. We bucket to low/mid/high so small numeric
changes don't produce jitter in the prompt.
"""

from __future__ import annotations

from typing import TypedDict

from app.models.character import Character


class StyleFragments(TypedDict):
    aggression: str
    risk_tolerance: str
    patience: str
    trash_talk: str


def _bucket(value: int) -> str:
    if value <= 3:
        return "low"
    if value <= 7:
        return "mid"
    return "high"


_AGGRESSION = {
    "low": "plays quietly and prefers to equalize before seeking chances",
    "mid": "plays solid chess but will seize initiative when it appears",
    "high": "plays aggressively, sacrificing material for initiative and the attack",
}

_RISK_TOLERANCE = {
    "low": "avoids speculative lines and declines unclear sacrifices",
    "mid": "takes calculated risks when the position justifies them",
    "high": "embraces sharp, double-edged positions even when the evaluation is murky",
}

_PATIENCE = {
    "low": "gets restless in slow positions and forces action, sometimes prematurely",
    "mid": "is comfortable with either maneuvering or sudden breakthroughs",
    "high": "will happily maneuver for dozens of moves, accumulating tiny advantages",
}

_TRASH_TALK = {
    "low": "speaks rarely and lets the moves do the talking",
    "mid": "chats warmly and comments on the game without needling the opponent",
    "high": "needles the opponent with sharp, confident remarks and enjoys the mind games",
}


def style_to_prompt_fragments(
    character: Character,
    *,
    slider_override: dict[str, int] | None = None,
) -> StyleFragments:
    """Return a dict of prompt snippets keyed by slider name.

    Pure function — does not read the DB or mutate the character.

    Phase 4.3 — an optional `slider_override` dict (from
    `evolution.effective_sliders(character, state)`) replaces the base
    values. Callers that haven't loaded an evolution state pass nothing
    and behaviour is unchanged.
    """
    overrides = slider_override or {}

    def pick(name: str) -> int:
        return int(overrides.get(name, getattr(character, name)))

    return StyleFragments(
        aggression=_AGGRESSION[_bucket(pick("aggression"))],
        risk_tolerance=_RISK_TOLERANCE[_bucket(pick("risk_tolerance"))],
        patience=_PATIENCE[_bucket(pick("patience"))],
        trash_talk=_TRASH_TALK[_bucket(pick("trash_talk"))],
    )


def style_summary_line(character: Character) -> str:
    """One-line summary usable as a compact prompt prefix."""
    f = style_to_prompt_fragments(character)
    return (
        f"{character.name} {f['aggression']}, {f['risk_tolerance']}, "
        f"{f['patience']}, and {f['trash_talk']}."
    )
