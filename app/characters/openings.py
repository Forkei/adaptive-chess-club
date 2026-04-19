"""Curated opening list.

Phase 1 only uses these for the character creation form and for storing
preferences on the character row. Phase 2's Director will read the
`group` tag to bias opening choice based on slider values (e.g. high
aggression + adaptive skill → prefer the `gambit` group).
"""

from __future__ import annotations

from typing import Literal, TypedDict

OpeningGroup = Literal[
    "king_pawn_open",
    "queen_pawn_closed",
    "flank",
    "gambit",
    "indian",
    "hypermodern",
    "unorthodox",
]


class Opening(TypedDict):
    eco: str
    name: str
    group: OpeningGroup


OPENINGS: list[Opening] = [
    # --- King's pawn, open games ---
    {"eco": "C20", "name": "King's Pawn Game", "group": "king_pawn_open"},
    {"eco": "C44", "name": "Scotch Game", "group": "king_pawn_open"},
    {"eco": "C50", "name": "Italian Game", "group": "king_pawn_open"},
    {"eco": "C55", "name": "Two Knights Defense", "group": "king_pawn_open"},
    {"eco": "C60", "name": "Ruy Lopez", "group": "king_pawn_open"},
    {"eco": "C42", "name": "Petrov Defense", "group": "king_pawn_open"},
    {"eco": "B01", "name": "Scandinavian Defense", "group": "king_pawn_open"},
    {"eco": "B20", "name": "Sicilian Defense", "group": "king_pawn_open"},
    {"eco": "B40", "name": "Sicilian Najdorf", "group": "king_pawn_open"},
    {"eco": "B70", "name": "Sicilian Dragon", "group": "king_pawn_open"},
    {"eco": "C00", "name": "French Defense", "group": "king_pawn_open"},
    {"eco": "B10", "name": "Caro-Kann Defense", "group": "king_pawn_open"},
    {"eco": "C40", "name": "Latvian Gambit", "group": "gambit"},

    # --- Queen's pawn, closed games ---
    {"eco": "D00", "name": "Queen's Pawn Game", "group": "queen_pawn_closed"},
    {"eco": "D06", "name": "Queen's Gambit", "group": "queen_pawn_closed"},
    {"eco": "D20", "name": "Queen's Gambit Accepted", "group": "queen_pawn_closed"},
    {"eco": "D30", "name": "Queen's Gambit Declined", "group": "queen_pawn_closed"},
    {"eco": "D43", "name": "Semi-Slav Defense", "group": "queen_pawn_closed"},
    {"eco": "D10", "name": "Slav Defense", "group": "queen_pawn_closed"},
    {"eco": "E00", "name": "Catalan Opening", "group": "queen_pawn_closed"},
    {"eco": "D70", "name": "Grünfeld Defense", "group": "hypermodern"},

    # --- Flank openings ---
    {"eco": "A10", "name": "English Opening", "group": "flank"},
    {"eco": "A04", "name": "Reti Opening", "group": "flank"},
    {"eco": "A00", "name": "Larsen's Opening (1.b3)", "group": "flank"},
    {"eco": "A02", "name": "Bird's Opening", "group": "flank"},

    # --- Indian systems ---
    {"eco": "E60", "name": "King's Indian Defense", "group": "indian"},
    {"eco": "E20", "name": "Nimzo-Indian Defense", "group": "indian"},
    {"eco": "E12", "name": "Queen's Indian Defense", "group": "indian"},
    {"eco": "A45", "name": "Trompowsky Attack", "group": "indian"},
    {"eco": "A48", "name": "London System", "group": "queen_pawn_closed"},

    # --- Hypermodern ---
    {"eco": "B06", "name": "Modern Defense", "group": "hypermodern"},
    {"eco": "B07", "name": "Pirc Defense", "group": "hypermodern"},
    {"eco": "A40", "name": "Benoni Defense", "group": "hypermodern"},

    # --- Gambits (sharp / aggressive) ---
    {"eco": "C30", "name": "King's Gambit", "group": "gambit"},
    {"eco": "C21", "name": "Danish Gambit", "group": "gambit"},
    {"eco": "C51", "name": "Evans Gambit", "group": "gambit"},
    {"eco": "D08", "name": "Albin Counter-Gambit", "group": "gambit"},
    {"eco": "A52", "name": "Budapest Gambit", "group": "gambit"},
    {"eco": "D00", "name": "Blackmar-Diemer Gambit", "group": "gambit"},

    # --- Unorthodox / rare ---
    {"eco": "A00", "name": "Grob's Attack (1.g4)", "group": "unorthodox"},
    {"eco": "B00", "name": "Nimzowitsch Defense", "group": "unorthodox"},
]


_BY_NAME: dict[str, Opening] = {o["name"]: o for o in OPENINGS}
_BY_ECO: dict[str, Opening] = {}
for _o in OPENINGS:
    _BY_ECO.setdefault(_o["eco"], _o)  # first entry wins when ECO code collides


def by_name(name: str) -> Opening | None:
    return _BY_NAME.get(name)


def by_eco(eco: str) -> Opening | None:
    return _BY_ECO.get(eco)


def groups_for(names_or_ecos: list[str]) -> list[OpeningGroup]:
    """Resolve a list of opening identifiers (names or ECO codes) to their groups."""
    groups: list[OpeningGroup] = []
    for ident in names_or_ecos:
        op = _BY_NAME.get(ident) or _BY_ECO.get(ident)
        if op is not None:
            groups.append(op["group"])
    return groups


def all_group_names() -> list[OpeningGroup]:
    return [
        "king_pawn_open",
        "queen_pawn_closed",
        "flank",
        "gambit",
        "indian",
        "hypermodern",
        "unorthodox",
    ]


# --- Phase 2b: coarse opening classifier ----------------------------------
#
# The curated OPENINGS list above has names + ECO codes but no move
# sequences. Rather than expand it (significant manual data entry) or
# pull in a polyglot book dependency, we ship a compact heuristic
# classifier keyed by SAN prefix. This covers the overwhelming majority
# of common openings at the family level. Matches that miss fall
# through to "unknown" — a legitimate signal on its own.
#
# TODO Phase 3: add a real opening book (python-chess supports polyglot)
# for ECO-code-level precision if that becomes useful. The 1.x.y prefix
# matching here bottoms out at the family level.

_CLASSIFIER_TABLE: list[tuple[tuple[str, ...], str, str, OpeningGroup]] = [
    # (san_prefix_tuple, eco, name, group)
    (("e4", "e5"),           "C20", "King's Pawn Game",        "king_pawn_open"),
    (("e4", "c5"),           "B20", "Sicilian Defense",        "king_pawn_open"),
    (("e4", "e6"),           "C00", "French Defense",          "king_pawn_open"),
    (("e4", "c6"),           "B10", "Caro-Kann Defense",       "king_pawn_open"),
    (("e4", "d5"),           "B01", "Scandinavian Defense",    "king_pawn_open"),
    (("e4", "Nf6"),          "B02", "Alekhine Defense",        "hypermodern"),
    (("e4", "d6"),           "B07", "Pirc Defense",            "hypermodern"),
    (("e4", "g6"),           "B06", "Modern Defense",          "hypermodern"),
    (("d4", "d5"),           "D00", "Queen's Pawn Game",       "queen_pawn_closed"),
    (("d4", "Nf6"),          "E60", "Indian Defense",          "indian"),
    (("d4", "f5"),           "A80", "Dutch Defense",           "hypermodern"),
    (("d4", "e6"),           "A40", "Queen's Pawn, e6 Systems", "queen_pawn_closed"),
    (("d4", "g6"),           "A40", "Modern Defense vs d4",    "hypermodern"),
    (("c4",),                "A10", "English Opening",         "flank"),
    (("Nf3",),               "A04", "Reti Opening",            "flank"),
    (("b3",),                "A00", "Larsen's Opening",        "flank"),
    (("f4",),                "A02", "Bird's Opening",          "flank"),
    (("g3",),                "A00", "King's Fianchetto",       "flank"),
    (("g4",),                "A00", "Grob's Attack",           "unorthodox"),
    (("b4",),                "A00", "Polish Opening",          "unorthodox"),
    (("e4", "e5", "Nf3", "Nc6", "Bb5"),  "C60", "Ruy Lopez",         "king_pawn_open"),
    (("e4", "e5", "Nf3", "Nc6", "Bc4"),  "C50", "Italian Game",       "king_pawn_open"),
    (("e4", "e5", "f4"),                 "C30", "King's Gambit",      "gambit"),
    (("d4", "d5", "c4"),                 "D06", "Queen's Gambit",     "queen_pawn_closed"),
    (("d4", "Nf6", "c4", "g6"),          "E60", "King's Indian Defense", "indian"),
    (("d4", "Nf6", "c4", "e6"),          "E20", "Nimzo-Indian Defense",  "indian"),
    (("d4", "d5", "Nc3"),                "D00", "Blackmar-Diemer Gambit", "gambit"),
]


def classify_opening(san_moves: list[str]) -> dict[str, str | None]:
    """Classify an opening by its first few SAN half-moves.

    Returns `{"eco": str, "name": str, "group": str}` — with all three
    set to `"unknown"` if no pattern matches. The longest matching
    prefix wins, so "e4 e5 f4" classifies as King's Gambit (gambit) not
    King's Pawn Game.

    Not ECO-code-perfect — see note at top of _CLASSIFIER_TABLE. Good
    enough for style-feature aggregation across matches.
    """
    if not san_moves:
        return {"eco": "unknown", "name": "unknown", "group": "unknown"}

    best: tuple[tuple[str, ...], str, str, OpeningGroup] | None = None
    for entry in _CLASSIFIER_TABLE:
        prefix, _eco, _name, _group = entry
        if len(san_moves) < len(prefix):
            continue
        if tuple(san_moves[: len(prefix)]) == prefix:
            if best is None or len(prefix) > len(best[0]):
                best = entry

    if best is None:
        return {"eco": "unknown", "name": "unknown", "group": "unknown"}
    _, eco, name, group = best
    return {"eco": eco, "name": name, "group": group}
