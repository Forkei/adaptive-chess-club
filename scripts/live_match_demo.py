"""Live end-to-end demo run.

Plays a short match against a preset character, captures every Soul
response + every surfaced memory, runs post-match, and writes a
Markdown report with the chat samples + generated memories.

Requires:
- GEMINI_API_KEY in .env (or env)
- sentence-transformers installed (for embeddings)
- maia2 package + weights OR Stockfish binary (MockEngine fallback works
  but produces terrible chess; we still get voice/memory signal)

Usage:
    python -m scripts.live_match_demo --character archibald --moves 14 --out logs/live_demo.md
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import chess

from app.characters.memory_generator import generate_and_store
from app.characters.seed import seed_presets
from app.config import get_settings
from app.db import SessionLocal, init_db
from app.engine.registry import reset_engines_for_testing
from app.llm.client import get_llm_client
from app.matches import service
from app.models.character import Character, CharacterState
from app.models.match import MatchAnalysis, MatchStatus, Player
from app.models.memory import Memory
from app.post_match.processor import ProcessorConfig, process_match_post_game

logger = logging.getLogger(__name__)


PRESET_KEYS = {
    "archibald": "archibald_finch",
    "viktor": "viktor_petrov",
    "margot": "margot_lindqvist",
    "kenji": "kenji_sato",
}


@dataclass
class TurnSample:
    move_number: int
    engine_move_san: str
    board_phase: str
    mood_before: dict
    agent_chat: str | None
    emotion: str
    emotion_intensity: float
    surfaced_memories: list[dict]
    player_chat: str | None = None


@dataclass
class RunOutput:
    character_name: str
    match_id: str
    turns: list[TurnSample] = field(default_factory=list)
    final_result: str = ""
    player_outcome: str = ""
    elo_before: int = 0
    elo_after: int = 0
    elo_delta_applied: int = 0
    elo_breakdown: dict = field(default_factory=dict)
    generated_memories: list[dict] = field(default_factory=list)
    narrative_summary: str = ""
    critical_moments: list[dict] = field(default_factory=list)


def _ensure_character(preset_key: str) -> str:
    """Seed presets and ensure the target character has memories."""
    from sqlalchemy import select

    seed_presets(run_generation=False)
    with SessionLocal() as s:
        row = s.execute(
            select(Character).where(Character.preset_key == preset_key)
        ).scalar_one_or_none()
        if row is None:
            raise RuntimeError(f"Preset {preset_key} not found after seeding")
        character_id = row.id
        mem_count = len(row.memories)
        state = row.state

    if mem_count < 5 or state != CharacterState.READY:
        logger.info("Generating memories for %s (current state=%s, memories=%d)…", preset_key, state, mem_count)
        generate_and_store(character_id)
    return character_id


def _random_legal_move(board: chess.Board, rng: random.Random) -> str:
    """Return a playable move for the player side.

    Prefers captures and checks early-ish for more interesting games, falls
    back to a random legal move. Not strong chess — we just want the match
    to progress with some variety.
    """
    legals = list(board.legal_moves)
    if not legals:
        raise RuntimeError("No legal moves")
    # Slight bias: 30% captures if any, otherwise random.
    captures = [m for m in legals if board.is_capture(m)]
    if captures and rng.random() < 0.3:
        return rng.choice(captures).uci()
    return rng.choice(legals).uci()


def _player_chat_sometimes(move_index: int, rng: random.Random) -> str | None:
    """Occasional player chat lines — keeps the Soul prompt varied."""
    chats = {
        2: "Let's see how this goes.",
        5: "Tight opening.",
        8: "I think I have a plan here.",
        11: "You're quieter than I expected.",
    }
    return chats.get(move_index)


async def run_live_match(
    character_id: str,
    *,
    max_moves: int,
    rng_seed: int = 7,
) -> RunOutput:
    rng = random.Random(rng_seed)

    # Refresh LLM + settings caches so the real .env key is used.
    from app.config import get_settings as _s
    from app.llm.client import get_llm_client as _c

    _s.cache_clear()
    _c.cache_clear()

    with SessionLocal() as s:
        char = s.get(Character, character_id)
        char_name = char.name
        elo_before = char.current_elo

        from app.auth import generate_guest_username

        player = Player(username=generate_guest_username(), display_name="LiveDemo")
        s.add(player)
        s.commit()
        player_id = player.id

        # Character plays white so Soul runs on turn 1.
        match = service.create_match(
            s, character_id=character_id, player_id=player_id, player_color="black"
        )
        s.commit()
        match_id = match.id
        await service.start_match_play(s, match)
        s.commit()

    out = RunOutput(character_name=char_name, match_id=match_id, elo_before=elo_before)

    for i in range(max_moves):
        # Pull current board + engine move from the latest move row.
        with SessionLocal() as s:
            match = service.get_match(s, match_id)
            if match.status != MatchStatus.IN_PROGRESS:
                break
            fen = match.current_fen
            last_moves = sorted(match.moves, key=lambda m: m.move_number)
            last_agent_move = next((m for m in reversed(last_moves) if m.side.value == "white"), None)
            # Capture the just-landed Soul output for the LAST engine turn.
            if last_agent_move is not None and last_agent_move.agent_chat_after is not None:
                # Already captured when the /move response fired. Skip duplicating here.
                pass

        board = chess.Board(fen)
        try:
            uci = _random_legal_move(board, rng)
        except RuntimeError:
            break

        player_chat = _player_chat_sometimes(i, rng)

        with SessionLocal() as s:
            pmove, amove, agent_outcome = await service.apply_player_move(
                s, match_id=match_id, uci=uci, player_chat=player_chat
            )
            s.commit()

        # Record the turn sample using the returned agent_outcome.
        if amove is not None and agent_outcome is not None:
            out.turns.append(
                TurnSample(
                    move_number=amove.move_number,
                    engine_move_san=amove.san,
                    board_phase=_phase_for_move(amove.move_number),
                    mood_before=dict(amove.mood_snapshot or {}),
                    agent_chat=agent_outcome.soul.speak,
                    emotion=agent_outcome.soul.emotion,
                    emotion_intensity=agent_outcome.soul.emotion_intensity,
                    surfaced_memories=[
                        {
                            "memory_id": m.memory_id,
                            "narrative_text": m.narrative_text,
                            "retrieval_reason": m.retrieval_reason,
                            "score": m.score,
                            "from_cache": m.from_cache,
                        }
                        for m in agent_outcome.surfaced
                    ],
                    player_chat=player_chat,
                )
            )

        with SessionLocal() as s:
            match = service.get_match(s, match_id)
            if match.status != MatchStatus.IN_PROGRESS:
                break

    # Resign if still in progress — we want a terminal state to trigger post-match.
    with SessionLocal() as s:
        match = service.get_match(s, match_id)
        if match.status == MatchStatus.IN_PROGRESS:
            service.resign(s, match_id=match_id)
            s.commit()
        out.final_result = match.result.value if match.result else "unresolved"

    # Post-match processor — synchronous so we can capture output.
    logger.info("Running post-match processor for %s…", match_id)
    try:
        process_match_post_game(
            match_id,
            config=ProcessorConfig(run_engine_analysis=False, run_llm_steps=True),
        )
    except Exception:
        logger.exception("Post-match processor raised; continuing to capture partial state.")

    # Collect post-match state.
    from sqlalchemy import select as _select

    with SessionLocal() as s:
        match = service.get_match(s, match_id)
        char = s.get(Character, character_id)
        out.elo_after = char.current_elo
        out.player_outcome = service.player_outcome(match) or ""

        analysis = s.execute(
            _select(MatchAnalysis).where(MatchAnalysis.match_id == match_id)
        ).scalar_one_or_none()
        if analysis is not None:
            out.elo_delta_applied = analysis.elo_delta_applied or 0
            out.critical_moments = list(analysis.critical_moments or [])
            from app.post_match.elo_apply import compute_elo_delta

            comp = compute_elo_delta(
                match=match,
                analysis_moves=(analysis.engine_analysis or {}).get("moves") or [],
            )
            out.elo_breakdown = {
                "outcome": int(comp.outcome_delta),
                "move_quality": round(comp.move_quality_delta, 1),
                "raw": round(comp.elo_delta_raw, 1),
                "short_halved": comp.short_match_halved,
                "rage_quit": comp.rage_quit_skipped_quality,
            }
            if analysis.generated_memory_ids:
                mems = list(
                    s.execute(
                        _select(Memory).where(Memory.id.in_(list(analysis.generated_memory_ids)))
                    ).scalars()
                )
                out.generated_memories = [
                    {
                        "narrative_text": m.narrative_text,
                        "triggers": list(m.triggers or []),
                        "relevance_tags": list(m.relevance_tags or []),
                        "scope": m.scope.value,
                        "emotional_valence": float(m.emotional_valence),
                    }
                    for m in mems
                ]

        from app.models.match import OpponentProfile

        profile = s.execute(
            _select(OpponentProfile).where(
                OpponentProfile.character_id == character_id,
                OpponentProfile.player_id == player_id,
            )
        ).scalar_one_or_none()
        if profile is not None:
            out.narrative_summary = profile.narrative_summary or ""

    return out


def _phase_for_move(move_number: int) -> str:
    if move_number < 20:
        return "opening"
    if move_number < 40:
        return "middlegame"
    return "endgame"


def _write_report(out: RunOutput, path: Path) -> None:
    lines: list[str] = []
    lines.append(f"# Live match demo — {out.character_name}")
    lines.append("")
    lines.append(f"- Match id: `{out.match_id}`")
    lines.append(f"- Final result: **{out.final_result}** (player: {out.player_outcome})")
    lines.append(
        f"- Elo: {out.elo_before} → {out.elo_after} "
        f"(delta {'+' if out.elo_delta_applied >= 0 else ''}{out.elo_delta_applied})"
    )
    if out.elo_breakdown:
        lines.append(f"- Breakdown: {out.elo_breakdown}")
    lines.append("")

    lines.append("## Turn-by-turn")
    lines.append("")
    for t in out.turns:
        header = f"### Move {t.move_number} ({t.engine_move_san}, {t.board_phase})"
        lines.append(header)
        lines.append(f"- Mood: {t.mood_before}")
        lines.append(f"- Emotion: `{t.emotion}` (intensity {t.emotion_intensity:.2f})")
        if t.player_chat:
            lines.append(f"- **Player**: _{t.player_chat}_")
        if t.agent_chat:
            lines.append(f"- **{out.character_name}**: {t.agent_chat}")
        else:
            lines.append(f"- **{out.character_name}**: (silent)")
        if t.surfaced_memories:
            lines.append("- Memories surfaced:")
            for m in t.surfaced_memories:
                tag = " (cached)" if m["from_cache"] else ""
                lines.append(f"  - score={m['score']:.2f}{tag}: {m['retrieval_reason']}")
                lines.append(f"    > {m['narrative_text'][:180]}")
        lines.append("")

    lines.append("## Generated memories (post-match)")
    lines.append("")
    if out.generated_memories:
        for m in out.generated_memories:
            lines.append(f"**{m['scope']}** · valence {m['emotional_valence']:+.2f}")
            lines.append(f"> {m['narrative_text']}")
            lines.append(f"- Triggers: {', '.join(m['triggers'])}")
            lines.append(f"- Tags: {', '.join(m['relevance_tags'])}")
            lines.append("")
    else:
        lines.append("_(none generated)_")
        lines.append("")

    if out.narrative_summary:
        lines.append("## Opponent note")
        lines.append("")
        lines.append(f"> {out.narrative_summary}")
        lines.append("")

    if out.critical_moments:
        lines.append("## Critical moments")
        lines.append("")
        for cm in out.critical_moments:
            lines.append(f"- move {cm.get('move_number')}: {cm.get('label')}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote report to %s", path)


async def main_async() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--character", default="archibald", choices=list(PRESET_KEYS))
    ap.add_argument("--moves", type=int, default=14)
    ap.add_argument("--out", default="logs/live_demo.md")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    init_db()
    reset_engines_for_testing()

    settings = get_settings()
    if not settings.gemini_api_key or settings.gemini_api_key == "test-key-not-real":
        print("ERROR: GEMINI_API_KEY missing or is the test placeholder.", file=sys.stderr)
        sys.exit(2)

    # Smoke-test the LLM so we fail fast on auth issues.
    try:
        get_llm_client()
    except Exception as exc:
        print(f"ERROR: LLM client init failed: {exc}", file=sys.stderr)
        sys.exit(2)

    preset_key = PRESET_KEYS[args.character]
    character_id = _ensure_character(preset_key)

    out = await run_live_match(character_id, max_moves=args.moves, rng_seed=args.seed)

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_report(out, path)

    # Short console summary.
    print()
    print(f"=== LIVE DEMO: {out.character_name} ===")
    print(f"Result: {out.final_result} (player: {out.player_outcome})")
    print(f"Elo: {out.elo_before} -> {out.elo_after} ({out.elo_delta_applied:+d})")
    chatty_turns = [t for t in out.turns if t.agent_chat]
    print(f"Chatty turns: {len(chatty_turns)} / {len(out.turns)}")
    print(f"Memories generated: {len(out.generated_memories)}")
    if chatty_turns:
        print("\nSample chat:")
        for t in chatty_turns[:3]:
            print(f"  move {t.move_number} [{t.emotion}]: {t.agent_chat}")
    if out.generated_memories:
        print("\nFirst generated memory:")
        print(f"  {out.generated_memories[0]['narrative_text']}")
    print(f"\nReport: {path}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
