"""Tone-constraint prompt fragments, keyed by `ContentRating`.

These are injected into every LLM prompt that generates prose in a
character's voice (backstory memories, post-match memories,
narrative summaries, Soul chat). Keeping them in one place means the
three rating levels can be tuned once and applied consistently.
"""

from __future__ import annotations

from app.models.character import ContentRating

_FAMILY = """\
CONTENT RATING: FAMILY
- Kid-friendly tone. No swearing. No sexual content or innuendo. No slurs.
- No graphic violence. Chess metaphor is fine ("crush their position", "his king is
  doomed"), but keep it away from real-world harm ("I'll kill you" — not OK).
- No political or religious provocation; no taboo-pushing.
- Strong personalities are still encouraged — just keep the register clean.
"""

_MATURE = """\
CONTENT RATING: MATURE
- Adult tone allowed. Mild swearing is fine if it fits the voice ("damn", "bastard",
  "hell"). Adult humor, sardonic wit, drinking references, world-weariness — all fine.
- Register latitude is broad enough for e.g. gamer slang ("bro", "waifu" ironically),
  plummy pomposity, gruff old-school machismo.
- NOT allowed: slurs (racial, sexual, ableist), explicit sexual content, graphic
  violence. No sexualizing minors. No real-world harm instructions.
"""

_UNRESTRICTED = """\
CONTENT RATING: UNRESTRICTED
- The character author has accepted responsibility for the tone.
- You may use stronger language, sharper edges, more volatile personalities.
- You MUST still refuse: content that sexualizes minors, instructions enabling
  real-world harm, slurs used to attack, and anything your safety training flags.
- Keep it in-character. Shock value without voice is bad writing.
"""


def rating_prompt_fragment(rating: ContentRating) -> str:
    """Return the tone-constraint block for the given rating."""
    if rating == ContentRating.FAMILY:
        return _FAMILY
    if rating == ContentRating.MATURE:
        return _MATURE
    return _UNRESTRICTED
