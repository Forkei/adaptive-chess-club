"""End-to-end post-match processor with mocked engine + LLM.

Exercises the full 5-step pipeline against a real in-memory SQLite
database. Engine analysis is bypassed via `run_engine_analysis=False`
(we feed canned features via a narrower test in test_post_match_elo);
the memory/narrative LLMs are stubbed.
"""

from __future__ import annotations

from typing import Any

import chess
import pytest
from pydantic import BaseModel

from app.db import SessionLocal
from app.models.character import Character, CharacterState
from app.models.match import (
    Color,
    Match,
    MatchAnalysis,
    MatchAnalysisStatus,
    MatchResult,
    MatchStatus,
    Move,
    OpponentProfile,
    Player,
)
from app.models.memory import Memory, MemoryScope, MemoryType
from app.post_match.processor import ProcessorConfig, process_match_post_game


class _FakeLLM:
    """Returns canned structured responses for both LLM calls."""

    def __init__(self):
        self.calls = []

    def generate_structured(self, *, prompt, response_schema, response_adapter=None, **kwargs):
        self.calls.append({"tag": kwargs.get("call_tag", ""), "prompt": prompt})
        # Memory generation returns a list[_MatchMemory].
        if "post_match_mem" in (kwargs.get("call_tag") or ""):
            from app.post_match.memory_gen import _MatchMemory

            return [
                _MatchMemory(
                    scope=MemoryScope.MATCH_RECAP,
                    type=MemoryType.OBSERVATION,
                    emotional_valence=0.1,
                    triggers=["e4", "Sicilian Najdorf", "opening theory", "test-player"],
                    narrative_text=(
                        "A short test game. The opponent played the Najdorf with "
                        "confidence, but resigned when the position turned."
                    ),
                    relevance_tags=["opening", "sicilian"],
                ),
            ]
        # Narrative summary.
        if "post_match_narrative" in (kwargs.get("call_tag") or ""):
            from app.post_match.memory_gen import _NarrativeOut

            return _NarrativeOut(summary="They play the Najdorf; they get nervous in worse positions.")
        raise RuntimeError(f"Unexpected LLM call_tag: {kwargs.get('call_tag')}")


def _setup_completed_match() -> str:
    with SessionLocal() as s:
        char = Character(
            name="PM Char",
            aggression=5,
            risk_tolerance=5,
            patience=5,
            trash_talk=5,
            target_elo=1500,
            current_elo=1500,
            floor_elo=1400,
            max_elo=1800,
            adaptive=True,
            state=CharacterState.READY,
        )
        s.add(char)
        from app.auth import generate_guest_username

        player = Player(username=generate_guest_username(), display_name="Postmatch Tester")
        s.add(player)
        s.commit()

        match = Match(
            character_id=char.id,
            player_id=player.id,
            player_color=Color.WHITE,  # player white, char black
            status=MatchStatus.COMPLETED,
            result=MatchResult.BLACK_WIN,  # char wins
            initial_fen=chess.STARTING_FEN,
            current_fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            move_count=3,
            character_elo_at_start=1500,
        )
        s.add(match)
        s.commit()

        for i, (side, uci, san) in enumerate(
            [
                (Color.WHITE, "e2e4", "e4"),
                (Color.BLACK, "c7c5", "c5"),
                (Color.WHITE, "g1f3", "Nf3"),
            ],
            start=1,
        ):
            s.add(
                Move(
                    match_id=match.id,
                    move_number=i,
                    side=side,
                    uci=uci,
                    san=san,
                    fen_after="",
                )
            )
        # Seed a couple of existing memories for voice sampling.
        s.add(
            Memory(
                character_id=char.id,
                scope=MemoryScope.CHARACTER_LORE,
                type=MemoryType.OPINION,
                emotional_valence=0.2,
                triggers=["old rivals"],
                narrative_text="I remember a game I won once.",
                relevance_tags=["rival"],
            )
        )
        s.commit()
        return match.id


def test_processor_end_to_end_happy_path():
    match_id = _setup_completed_match()
    llm = _FakeLLM()

    process_match_post_game(
        match_id,
        llm=llm,
        config=ProcessorConfig(run_engine_analysis=False, run_llm_steps=True),
    )

    with SessionLocal() as s:
        analysis = s.execute(
            __import__("sqlalchemy").select(MatchAnalysis).where(MatchAnalysis.match_id == match_id)
        ).scalar_one()
        assert analysis.status == MatchAnalysisStatus.COMPLETED
        assert set(analysis.steps_completed) == {
            "engine_analysis",
            "feature_extraction",
            "elo_ratchet",
            "evolution",  # Phase 4.3 — pure-data step after Elo ratchet.
            "memory_generation",
            "narrative_summary",
        }
        assert analysis.elo_delta_applied is not None
        # Patch Pass 2 Item 2 math: char 1500 vs player 1200 default.
        # expected ≈ 0.849, actual=1.0, K=32 → outcome ≈ 4.83
        # move_count=3 < 10 → short-match scale ×0.3 → ≈ 1.45 → rounded 1.
        assert analysis.elo_delta_applied == 1
        # Player-side ratchet applied: player lost → small Elo loss.
        assert analysis.player_elo_delta_applied == -1
        # At least one memory generated.
        assert len(analysis.generated_memory_ids) == 1

    # Verify side-effects: character current_elo bumped, profile updated, memory persisted.
    with SessionLocal() as s:
        match = s.get(Match, match_id)
        char = s.get(Character, match.character_id)
        assert char.current_elo == 1501
        player = s.get(Player, match.player_id)
        assert player.elo == 1199  # lost 1 Elo

        profile = s.execute(
            __import__("sqlalchemy").select(OpponentProfile)
            .where(OpponentProfile.character_id == match.character_id)
            .where(OpponentProfile.player_id == match.player_id)
        ).scalar_one()
        assert profile.games_played == 1
        assert profile.games_won_by_character == 1
        assert profile.narrative_summary  # LLM produced one
        assert "Najdorf" in profile.narrative_summary

        new_mem = s.execute(
            __import__("sqlalchemy").select(Memory)
            .where(Memory.character_id == match.character_id)
            .where(Memory.scope == MemoryScope.MATCH_RECAP)
        ).scalar_one()
        assert new_mem.match_id == match_id
        assert new_mem.player_id == match.player_id
        assert len(new_mem.triggers) >= 3


def test_processor_is_idempotent():
    """Second run after completion is a no-op."""
    match_id = _setup_completed_match()
    llm = _FakeLLM()
    process_match_post_game(
        match_id,
        llm=llm,
        config=ProcessorConfig(run_engine_analysis=False, run_llm_steps=True),
    )
    first_calls = len(llm.calls)
    process_match_post_game(
        match_id,
        llm=llm,
        config=ProcessorConfig(run_engine_analysis=False, run_llm_steps=True),
    )
    assert len(llm.calls) == first_calls  # no further LLM traffic


def test_processor_rage_quit_outcome_only():
    with SessionLocal() as s:
        char = Character(
            name="Rage Receiver",
            aggression=5,
            risk_tolerance=5,
            patience=5,
            trash_talk=5,
            target_elo=1500,
            current_elo=1500,
            floor_elo=1400,
            max_elo=1800,
            adaptive=True,
            state=CharacterState.READY,
        )
        s.add(char)
        from app.auth import generate_guest_username

        player = Player(username=generate_guest_username(), display_name="Quitter")
        s.add(player)
        s.commit()
        match = Match(
            character_id=char.id,
            player_id=player.id,
            player_color=Color.WHITE,
            status=MatchStatus.ABANDONED,
            result=MatchResult.ABANDONED,
            initial_fen=chess.STARTING_FEN,
            current_fen=chess.STARTING_FEN,
            move_count=0,
            character_elo_at_start=1500,
        )
        s.add(match)
        s.commit()
        match_id = match.id

    llm = _FakeLLM()
    process_match_post_game(
        match_id,
        llm=llm,
        config=ProcessorConfig(run_engine_analysis=False, run_llm_steps=False),
    )

    with SessionLocal() as s:
        analysis = s.execute(
            __import__("sqlalchemy").select(MatchAnalysis).where(MatchAnalysis.match_id == match_id)
        ).scalar_one()
        # Patch Pass 2 Item 2: rage quit uses expected-score math but skips
        # move quality AND the short-match scale. char 1500 vs player 1200,
        # expected ≈ 0.849, actual=1.0, K=32 → raw ≈ 4.83, applied 5.
        assert analysis.elo_delta_raw > 4.5 and analysis.elo_delta_raw < 5.5
        assert analysis.elo_delta_applied == 5

        char = s.get(Character, match.character_id)
        assert char.current_elo == 1505


def test_resigned_is_not_rage_quit():
    """A RESIGNED match runs the full Elo math (not outcome-only)
    and hits the respectful memory-gen branch, not the rage-quit one."""
    from app.post_match.memory_gen import _outcome_phrase, _build_memory_prompt

    with SessionLocal() as s:
        char = Character(
            name="Polite Winner",
            aggression=5, risk_tolerance=5, patience=5, trash_talk=5,
            target_elo=1500, current_elo=1500, floor_elo=1400, max_elo=1800,
            adaptive=True, state=CharacterState.READY,
        )
        s.add(char)
        from app.auth import generate_guest_username

        player = Player(username=generate_guest_username(), display_name="Resigner")
        s.add(player)
        s.commit()
        match = Match(
            character_id=char.id,
            player_id=player.id,
            player_color=Color.WHITE,
            # Player was white; character (black) won via resign.
            status=MatchStatus.RESIGNED,
            result=MatchResult.BLACK_WIN,
            initial_fen=chess.STARTING_FEN,
            current_fen=chess.STARTING_FEN,
            move_count=20,
            character_elo_at_start=1500,
        )
        s.add(match)
        s.commit()

        # Prompt branch: respectful, not rage-quit framing.
        phrase = _outcome_phrase(match)
        assert "rage quit" not in phrase.lower()
        assert "resigned" in phrase.lower()

        prompt = _build_memory_prompt(
            character=char, match=match,
            critical_moments=[], features_before=None, features_after={},
            opponent_notes=[], prior_memory_samples=[],
        )
        # Respectful branch should be present; the rage-quit instruction block
        # ("because the player rage-quit, ...") should NOT be.
        assert "clean concession" in prompt.lower() or "legitimate finish" in prompt.lower()
        assert "because the player rage-quit" not in prompt.lower()


def test_processor_handles_memory_llm_failure_gracefully():
    """Memory LLM failure → memory step fails; elo+features still succeed."""

    class _FailingLLM:
        def generate_structured(self, **kwargs):
            from app.llm.client import LLMError

            raise LLMError("simulated outage")

    match_id = _setup_completed_match()
    process_match_post_game(
        match_id,
        llm=_FailingLLM(),
        config=ProcessorConfig(run_engine_analysis=False, run_llm_steps=True),
    )

    with SessionLocal() as s:
        analysis = s.execute(
            __import__("sqlalchemy").select(MatchAnalysis).where(MatchAnalysis.match_id == match_id)
        ).scalar_one()
        # Processor finalizes as completed — LLM outages degrade gracefully
        # (memory_gen returns empty list, narrative leaves profile unchanged)
        # rather than marking the whole pipeline FAILED.
        assert analysis.status == MatchAnalysisStatus.COMPLETED
        assert "elo_ratchet" in analysis.steps_completed
        # The LLM outage left no memories and no narrative update.
        assert analysis.generated_memory_ids == []
