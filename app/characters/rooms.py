"""Phase 4.1 — per-character "room" metadata.

Each preset grandmaster gets a themed room: a palette (CSS-variable
overrides that apply to the detail + play pages), a tagline, an ambient
audio bed, a background image/video, and a set of emotion video clips
keyed by the same strings the Soul emits (`neutral`, `pleased`, …).

Why this lives outside `Character` ORM:
- The theme is a property of the preset *kind*, not an instance. Custom
  player-authored characters fall back to `DEFAULT_ROOM`.
- Keeps the DB migration story clean. Adding a character-level override
  later would be a single JSON column; not needed for 4.1.

Asset conventions (see `app/web/static/characters/README.md`):
    app/web/static/characters/<preset_key>/
        ambient.mp3                 # looped ambient bed
        background.{webp,jpg,mp4}   # room backdrop
        emotions/<emotion>.mp4      # per-emotion clip
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RoomTheme:
    """Declarative per-character room. Rendered into the page as CSS
    variable overrides + data attributes + asset URLs.
    """

    slug: str
    display_name: str
    tagline: str
    # CSS custom-property overrides. Only touch "mood" tokens; functional
    # tokens (rating chips, conn-pill states) stay global.
    css_vars: dict[str, str] = field(default_factory=dict)
    # Path relative to /static, or None.
    ambient_track: str | None = None
    background: str | None = None
    background_kind: str = "none"  # "image" | "video" | "none"
    # Soul-emitted emotion → clip URL (relative to /static). Missing
    # emotions fall back to `neutral` if present; if neutral is absent
    # too, the video element stays hidden and the existing text
    # indicator takes over.
    emotion_clips: dict[str, str] = field(default_factory=dict)


# Soul emits these eight emotions (see app/schemas/agents.py Emotion literal).
EMOTIONS: tuple[str, ...] = (
    "neutral",
    "pleased",
    "annoyed",
    "excited",
    "focused",
    "uncertain",
    "smug",
    "deflated",
)


def _viktor_clips() -> dict[str, str]:
    """Viktor is the only preset with real assets so far. Each Soul
    emotion points at a filename; the user drops MP4s into
    /static/characters/viktor_petrov/emotions/ with matching names.

    Missing files → the emotion_video.js module silently falls back to
    the neutral clip, then to the text-only indicator.
    """
    base = "/static/characters/viktor_petrov/emotions"
    return {e: f"{base}/{e}.mp4" for e in EMOTIONS}


def _kenji_clips() -> dict[str, str]:
    """Kenji emotion clips shipped as MP4s under
    /static/characters/kenji_sato/emotions/. Mapped from the supplied
    reel so every Soul emotion has a reasonable visual match."""
    base = "/static/characters/kenji_sato/emotions"
    return {e: f"{base}/{e}.mp4" for e in EMOTIONS}


VIKTOR_ROOM = RoomTheme(
    slug="viktor_petrov",
    display_name="Viktor Petrov",
    tagline="Above the bakery · Brooklyn, 23:14",
    css_vars={
        "--mp-bg":          "#0F0905",
        "--mp-surface-1":   "#1A120C",
        "--mp-surface-2":   "#231811",
        "--mp-surface-3":   "#2C1F17",
        "--mp-hairline":    "#3A2619",
        "--mp-hairline-2":  "#4A3022",
        "--mp-brass":       "#C58147",
        "--mp-brass-bright":"#E0995B",
        "--mp-brass-dim":   "#8A5A30",
        "--mp-oxblood":     "#B53A36",
    },
    # No recorded track shipped — falls through to synth_ambient.js which
    # generates a room-specific pad in the browser. Set to a real path to
    # switch to file playback.
    ambient_track=None,
    background="/static/characters/viktor_petrov/background.webp",
    background_kind="image",
    emotion_clips=_viktor_clips(),
)


MARGOT_ROOM = RoomTheme(
    slug="margot_lindqvist",
    display_name="Margot Lindqvist",
    tagline="The honey-room · Outside Lund, dusk",
    css_vars={
        "--mp-bg":          "#0B1310",
        "--mp-surface-1":   "#131C17",
        "--mp-surface-2":   "#1C2620",
        "--mp-surface-3":   "#253028",
        "--mp-hairline":    "#2B3730",
        "--mp-hairline-2":  "#3A4A40",
        "--mp-brass":       "#A8B07A",
        "--mp-brass-bright":"#C3CA97",
        "--mp-brass-dim":   "#767E52",
        "--mp-felt":        "#3C7F6F",
        "--mp-felt-bright": "#6FC0A8",
    },
    ambient_track=None,
    background="/static/characters/margot_lindqvist/background.webp",
    background_kind="image",
    emotion_clips={},
)


KENJI_ROOM = RoomTheme(
    slug="kenji_sato",
    display_name="Kenji 'Lightning' Sato",
    tagline="Stream on · Yokohama, past midnight",
    css_vars={
        "--mp-bg":          "#0A0814",
        "--mp-surface-1":   "#120F20",
        "--mp-surface-2":   "#1A1530",
        "--mp-surface-3":   "#241D3F",
        "--mp-hairline":    "#2A2347",
        "--mp-hairline-2":  "#3B3160",
        "--mp-brass":       "#E25BA8",
        "--mp-brass-bright":"#F37EC0",
        "--mp-brass-dim":   "#9E3E73",
        "--mp-ink-blue":    "#7AA7FF",
        "--mp-ink-blue-alt":"#A8C3FF",
    },
    ambient_track="/static/audio/ambient/coffee_shop.ogg",
    background="/static/characters/kenji_sato/background.webp",
    background_kind="image",
    emotion_clips=_kenji_clips(),
)


ARCHIBALD_ROOM = RoomTheme(
    slug="archibald_finch",
    display_name="Professor Archibald Finch",
    tagline="The corner table at the club · West London, rainy",
    css_vars={
        "--mp-bg":          "#120E08",
        "--mp-surface-1":   "#1C170E",
        "--mp-surface-2":   "#261F13",
        "--mp-surface-3":   "#2F271A",
        "--mp-hairline":    "#3A301E",
        "--mp-hairline-2":  "#4B3F2A",
        "--mp-brass":       "#D4B585",
        "--mp-brass-bright":"#E8CDA3",
        "--mp-brass-dim":   "#8F7A54",
    },
    ambient_track=None,
    background="/static/characters/archibald_finch/background.webp",
    background_kind="image",
    emotion_clips={},
)


DEFAULT_ROOM = RoomTheme(
    slug="default",
    display_name="",
    tagline="A visiting player's table",
    css_vars={},
    ambient_track=None,
    background=None,
    background_kind="none",
    emotion_clips={},
)


_BY_PRESET: dict[str, RoomTheme] = {
    VIKTOR_ROOM.slug: VIKTOR_ROOM,
    MARGOT_ROOM.slug: MARGOT_ROOM,
    KENJI_ROOM.slug: KENJI_ROOM,
    ARCHIBALD_ROOM.slug: ARCHIBALD_ROOM,
}


def theme_for_preset_key(preset_key: str | None) -> RoomTheme:
    if preset_key and preset_key in _BY_PRESET:
        return _BY_PRESET[preset_key]
    return DEFAULT_ROOM


def theme_for_character(character) -> RoomTheme:
    """`character` is a Character ORM row or anything with a `preset_key`
    attribute; non-preset custom characters → DEFAULT_ROOM.
    """
    return theme_for_preset_key(getattr(character, "preset_key", None))
