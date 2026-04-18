"""Ground-truth board description for the Soul prompt.

LLMs cannot reliably read a FEN. They can confidently invent pieces,
miscount material, or overlook a pin. The only safe move is to never
put a FEN in front of the Soul — feed it a `BoardSummary` derived
programmatically from the board via python-chess.

The structured fields are what downstream code reads; the `prose`
block is what the Soul prompt will include verbatim in Phase 2b.
"""

from __future__ import annotations

from typing import Literal

import chess
from pydantic import BaseModel, Field

Phase = Literal["opening", "middlegame", "endgame"]
PawnStructure = Literal["open", "semi_open", "closed", "dynamic"]


PIECE_VALUE = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


class MaterialSide(BaseModel):
    pawns: int
    knights: int
    bishops: int
    rooks: int
    queens: int
    total_value: int  # points (pawn = 1)


class BoardSummary(BaseModel):
    fen: str
    move_number: int
    side_to_move: Literal["white", "black"]
    phase: Phase

    white_material: MaterialSide
    black_material: MaterialSide
    material_delta: int  # white points - black points

    white_castling: Literal["both", "kingside", "queenside", "none"]
    black_castling: Literal["both", "kingside", "queenside", "none"]

    central_pawn_structure: PawnStructure
    central_pawns: list[str]  # occupants on d4/d5/e4/e5

    white_in_check: bool
    black_in_check: bool

    # Lightweight tactical flags
    pinned_pieces: list[str] = Field(default_factory=list)       # "d7 bishop pinned against king"
    hanging_pieces: list[str] = Field(default_factory=list)       # "f6 knight attacked, undefended"
    king_safety_concerns: list[str] = Field(default_factory=list)

    # Optional eval
    eval_cp: int | None = None
    eval_prose: str | None = None  # "White has a slight edge (+0.4)"

    prose: str  # multi-line ground-truth summary the Soul can quote from


def _count_material(board: chess.Board, color: chess.Color) -> MaterialSide:
    counts = {pt: 0 for pt in PIECE_VALUE}
    for piece in board.piece_map().values():
        if piece.color == color:
            counts[piece.piece_type] += 1
    total = sum(n * PIECE_VALUE[pt] for pt, n in counts.items())
    return MaterialSide(
        pawns=counts[chess.PAWN],
        knights=counts[chess.KNIGHT],
        bishops=counts[chess.BISHOP],
        rooks=counts[chess.ROOK],
        queens=counts[chess.QUEEN],
        total_value=total,
    )


def _castling_status(board: chess.Board, color: chess.Color) -> Literal["both", "kingside", "queenside", "none"]:
    k = board.has_kingside_castling_rights(color)
    q = board.has_queenside_castling_rights(color)
    if k and q:
        return "both"
    if k:
        return "kingside"
    if q:
        return "queenside"
    return "none"


def _central_pawn_structure(board: chess.Board) -> tuple[PawnStructure, list[str]]:
    center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
    occupants: list[str] = []
    pawn_occupants = 0
    for sq in center_squares:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        color = "white" if piece.color == chess.WHITE else "black"
        occupants.append(f"{chess.square_name(sq)}={color}-{chess.piece_name(piece.piece_type)}")
        if piece.piece_type == chess.PAWN:
            pawn_occupants += 1

    if pawn_occupants == 0:
        structure: PawnStructure = "open"
    elif pawn_occupants == 4:
        structure = "closed"
    elif pawn_occupants == 1:
        structure = "semi_open"
    else:
        structure = "dynamic"
    return structure, occupants


def _phase(board: chess.Board) -> Phase:
    """Endgame when non-pawn material is low; otherwise opening <10 moves, else middlegame."""
    non_pawn_value = 0
    for piece in board.piece_map().values():
        if piece.piece_type == chess.PAWN or piece.piece_type == chess.KING:
            continue
        non_pawn_value += PIECE_VALUE[piece.piece_type]
    if non_pawn_value <= 16:  # roughly: each side has ~rook + minor + queen or less
        return "endgame"
    if board.fullmove_number < 10:
        return "opening"
    return "middlegame"


def _pinned_pieces_for(board: chess.Board, color: chess.Color) -> list[str]:
    """Return 'e7 knight pinned against king' style strings."""
    out: list[str] = []
    for square, piece in board.piece_map().items():
        if piece.color != color or piece.piece_type == chess.KING:
            continue
        if board.is_pinned(color, square):
            color_label = "white" if color == chess.WHITE else "black"
            out.append(
                f"{chess.square_name(square)} {color_label} {chess.piece_name(piece.piece_type)} "
                f"pinned against king"
            )
    return out


def _hanging_pieces_for(board: chess.Board, color: chess.Color) -> list[str]:
    """Undefended pieces being attacked by a piece of lower or equal value."""
    out: list[str] = []
    opp = not color
    for square, piece in board.piece_map().items():
        if piece.color != color or piece.piece_type == chess.KING:
            continue
        attackers = board.attackers(opp, square)
        if not attackers:
            continue
        defenders = board.attackers(color, square)
        if defenders:
            continue
        # Any attacker of not-larger value is a win-of-material threat.
        min_attacker_value = min(
            PIECE_VALUE[board.piece_at(a).piece_type] for a in attackers  # type: ignore[union-attr]
        )
        if min_attacker_value <= PIECE_VALUE[piece.piece_type]:
            color_label = "white" if color == chess.WHITE else "black"
            out.append(
                f"{chess.square_name(square)} {color_label} {chess.piece_name(piece.piece_type)}"
                f" attacked and undefended"
            )
    return out


def _king_safety_for(board: chess.Board, color: chess.Color) -> list[str]:
    king_sq = board.king(color)
    if king_sq is None:
        return []
    out: list[str] = []
    attackers = board.attackers(not color, king_sq)
    if attackers:
        color_label = "white" if color == chess.WHITE else "black"
        out.append(f"{color_label} king directly attacked from {len(attackers)} square(s)")
    # Heuristic: weakened pawn shield around castled king
    if board.has_castling_rights(color) is False and king_sq is not None:
        shield_offsets = [(0, 1), (-1, 1), (1, 1)] if color == chess.WHITE else [(0, -1), (-1, -1), (1, -1)]
        missing = 0
        kf, kr = chess.square_file(king_sq), chess.square_rank(king_sq)
        for df, dr in shield_offsets:
            tf, tr = kf + df, kr + dr
            if 0 <= tf < 8 and 0 <= tr < 8:
                sq = chess.square(tf, tr)
                p = board.piece_at(sq)
                if p is None or p.piece_type != chess.PAWN or p.color != color:
                    missing += 1
        if missing >= 2:
            color_label = "white" if color == chess.WHITE else "black"
            out.append(f"{color_label} king has a weakened pawn shield")
    return out


def _eval_prose(eval_cp: int | None) -> str | None:
    if eval_cp is None:
        return None
    pawns = eval_cp / 100.0
    if abs(eval_cp) >= 100000:
        return "Forced mate on the board"
    if abs(pawns) < 0.3:
        return "The position is roughly balanced"
    side = "White" if pawns > 0 else "Black"
    magnitude = abs(pawns)
    if magnitude < 0.8:
        desc = "a slight edge"
    elif magnitude < 1.8:
        desc = "a meaningful advantage"
    elif magnitude < 3.5:
        desc = "a significant advantage"
    else:
        desc = "a winning advantage"
    return f"{side} has {desc} ({'+' if pawns > 0 else ''}{pawns:.1f})"


def _prose(summary_fields: dict) -> str:
    lines: list[str] = []
    lines.append(
        f"Move {summary_fields['move_number']}, {summary_fields['side_to_move']} to move; "
        f"{summary_fields['phase']}."
    )
    lines.append(
        f"Material: W {summary_fields['white_material'].total_value} — "
        f"B {summary_fields['black_material'].total_value} "
        f"(delta {summary_fields['material_delta']:+d})."
    )
    lines.append(
        f"Castling rights: white {summary_fields['white_castling']}, "
        f"black {summary_fields['black_castling']}."
    )
    lines.append(f"Central pawn structure: {summary_fields['central_pawn_structure']}.")
    if summary_fields["white_in_check"]:
        lines.append("White is in check.")
    if summary_fields["black_in_check"]:
        lines.append("Black is in check.")
    if summary_fields["pinned_pieces"]:
        lines.append("Pins: " + "; ".join(summary_fields["pinned_pieces"]))
    if summary_fields["hanging_pieces"]:
        lines.append("Hanging: " + "; ".join(summary_fields["hanging_pieces"]))
    if summary_fields["king_safety_concerns"]:
        lines.append("King safety: " + "; ".join(summary_fields["king_safety_concerns"]))
    if summary_fields["eval_prose"]:
        lines.append(summary_fields["eval_prose"])
    return "\n".join(lines)


def board_to_english(board: chess.Board, *, eval_cp: int | None = None) -> BoardSummary:
    """Build a ground-truth `BoardSummary` for the given position."""
    white_mat = _count_material(board, chess.WHITE)
    black_mat = _count_material(board, chess.BLACK)
    structure, occupants = _central_pawn_structure(board)

    pinned = _pinned_pieces_for(board, chess.WHITE) + _pinned_pieces_for(board, chess.BLACK)
    hanging = _hanging_pieces_for(board, chess.WHITE) + _hanging_pieces_for(board, chess.BLACK)
    king_concerns = _king_safety_for(board, chess.WHITE) + _king_safety_for(board, chess.BLACK)

    fields = {
        "fen": board.fen(),
        "move_number": board.fullmove_number,
        "side_to_move": "white" if board.turn == chess.WHITE else "black",
        "phase": _phase(board),
        "white_material": white_mat,
        "black_material": black_mat,
        "material_delta": white_mat.total_value - black_mat.total_value,
        "white_castling": _castling_status(board, chess.WHITE),
        "black_castling": _castling_status(board, chess.BLACK),
        "central_pawn_structure": structure,
        "central_pawns": occupants,
        "white_in_check": board.is_check() and board.turn == chess.WHITE,
        "black_in_check": board.is_check() and board.turn == chess.BLACK,
        "pinned_pieces": pinned,
        "hanging_pieces": hanging,
        "king_safety_concerns": king_concerns,
        "eval_cp": eval_cp,
        "eval_prose": _eval_prose(eval_cp),
    }
    fields["prose"] = _prose(fields)
    return BoardSummary(**fields)
