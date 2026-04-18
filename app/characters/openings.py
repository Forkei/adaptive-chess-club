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
