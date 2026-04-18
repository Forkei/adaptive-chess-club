from __future__ import annotations

import chess

from app.engine.board_abstraction import board_to_english


def test_starting_position_phase_and_material():
    s = board_to_english(chess.Board())
    assert s.phase == "opening"
    assert s.white_material.total_value == 39  # 8+6+6+10+9
    assert s.black_material.total_value == 39
    assert s.material_delta == 0
    assert s.white_castling == "both"
    assert s.black_castling == "both"
    assert s.central_pawn_structure == "open"
    assert s.side_to_move == "white"
    assert not s.white_in_check and not s.black_in_check


def test_e4_opens_with_one_central_pawn():
    b = chess.Board()
    b.push_uci("e2e4")
    s = board_to_english(b)
    assert s.central_pawn_structure == "semi_open"
    assert any("e4" in cp for cp in s.central_pawns)


def test_pawn_storm_closed_center():
    # Manually build a closed center: pawns on d4 d5 e4 e5.
    b = chess.Board()
    for move in ["d2d4", "d7d5", "e2e3"]:
        b.push_uci(move)
    # Need a black pawn on e5. Construct instead via FEN.
    b = chess.Board("rnbqkbnr/ppp2ppp/8/3pp3/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 3")
    s = board_to_english(b)
    assert s.central_pawn_structure == "closed"


def test_detects_pinned_piece():
    # Set up a simple pin: Black knight on d7 pinned by white bishop on g4 via
    # the e8 king along the diagonal? Actually: Black king e8, Black piece e7,
    # white rook e1. The e7 piece is pinned along the e-file.
    b = chess.Board("4k3/4r3/8/8/8/8/8/4R2K w - - 0 1")
    # Black rook on e7 pinned against king on e8 by white rook on e1.
    s = board_to_english(b)
    assert any("pinned" in note for note in s.pinned_pieces)


def test_detects_hanging_piece():
    # Black knight on e5 attacked by white pawn d4 and undefended.
    b = chess.Board("4k3/8/8/4n3/3P4/8/8/4K3 w - - 0 1")
    s = board_to_english(b)
    assert any("e5" in note and "attacked" in note for note in s.hanging_pieces)


def test_endgame_phase_detected_when_material_low():
    # K+R vs K.
    b = chess.Board("4k3/8/8/8/8/8/4R3/4K3 w - - 0 1")
    s = board_to_english(b)
    assert s.phase == "endgame"


def test_eval_prose_rendered_with_sign():
    s = board_to_english(chess.Board(), eval_cp=45)
    assert s.eval_prose is not None
    assert "White" in s.eval_prose or "slight" in s.eval_prose.lower()

    s2 = board_to_english(chess.Board(), eval_cp=-400)
    assert s2.eval_prose is not None
    assert "Black" in s2.eval_prose
    assert "winning" in s2.eval_prose.lower() or "significant" in s2.eval_prose.lower()


def test_prose_is_multiline_and_mentions_core_facts():
    s = board_to_english(chess.Board())
    assert "\n" in s.prose
    assert "Move 1" in s.prose
    assert "Material" in s.prose
