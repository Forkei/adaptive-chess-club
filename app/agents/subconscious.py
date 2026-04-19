"""The Subconscious agent.

Called before the Soul on every character turn. Picks up to 5 memories
that are most resonant with the current moment using a weighted blend
of semantic similarity, trigger overlap, opponent relevance, mood
alignment, and recency penalty.

If the top-1 score is decisively above top-2 (margin > 0.15), the
top-5 are returned directly. Otherwise the top-8 are handed to a Flash
Lite re-rank step (structured output) to pick the best 5 with
retrieval reasons.

Per-match caching: a cache entry is keyed by (last_player_uci,
last_chat_hash, mood_polarity_bucket). If the same key fires within 3
character turns, the previously-surfaced list is returned with
`from_cache=True` — we don't re-run the pipeline (saves the LLM call
AND the embedding call).
"""

from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, TypeAdapter
from sqlalchemy.orm import Session

from app.agents.retrieval import (
    DEFAULT_WEIGHTS,
    ScoreBreakdown,
    aggregate_scores,
    build_context_tokens,
    mood_alignment_score,
    mood_polarity_bucket,
    opponent_relevance_score,
    recency_penalty,
    trigger_match_score,
)
from app.director.mood import MoodState
from app.engine.board_abstraction import BoardSummary
from app.llm.client import LLMClient, LLMError, get_llm_client
from app.memory import vector_store
from app.memory.crud import get_by_ids
from app.memory.embeddings import embed_text
from app.models.character import Character
from app.models.memory import Memory
from app.schemas.agents import SurfacedMemory

logger = logging.getLogger(__name__)

# --- Tuning constants ------------------------------------------------------

CANDIDATE_POOL = 50            # pre-filter from semantic top-k
TOP_K_FINAL = 5                # memories returned to Soul
TOP_K_FOR_RERANK = 8           # candidates sent to LLM when margin is tight
LLM_RERANK_MARGIN = 0.15       # top_1 - top_2 threshold for skipping LLM
CACHE_TTL_TURNS = 3            # max turns a cache entry is valid


# --- Cache -----------------------------------------------------------------


@dataclass
class _CacheEntry:
    cache_key: str
    turn: int  # move_number when computed
    surfaced: list[SurfacedMemory]


_CACHE: dict[str, _CacheEntry] = {}
_CACHE_LOCK = threading.Lock()


def _mood_chat_hash(text: str | None) -> str:
    if not text:
        return "none"
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:10]


def build_cache_key(
    *,
    last_player_uci: str | None,
    last_player_chat: str | None,
    mood: MoodState,
) -> str:
    return f"{last_player_uci or 'none'}|{_mood_chat_hash(last_player_chat)}|{mood_polarity_bucket(mood)}"


def _read_cache(match_id: str, cache_key: str, current_turn: int) -> list[SurfacedMemory] | None:
    with _CACHE_LOCK:
        entry = _CACHE.get(match_id)
    if entry is None:
        return None
    if entry.cache_key != cache_key:
        return None
    if current_turn - entry.turn >= CACHE_TTL_TURNS:
        return None
    # Return copies marked from_cache=True.
    return [sm.model_copy(update={"from_cache": True}) for sm in entry.surfaced]


def _write_cache(match_id: str, cache_key: str, current_turn: int, surfaced: list[SurfacedMemory]) -> None:
    with _CACHE_LOCK:
        _CACHE[match_id] = _CacheEntry(
            cache_key=cache_key,
            turn=current_turn,
            surfaced=[sm.model_copy() for sm in surfaced],
        )


def clear_cache(match_id: str | None = None) -> None:
    """Test helper / match-end cleanup."""
    with _CACHE_LOCK:
        if match_id is None:
            _CACHE.clear()
        else:
            _CACHE.pop(match_id, None)


# --- Input / dependencies --------------------------------------------------


@dataclass
class SubconsciousInput:
    """Everything the Subconscious needs for one call.

    Keeping it a dataclass (not a dependency on the match Session) means
    tests can exercise the pipeline with fabricated inputs + an in-memory
    SQLAlchemy session.
    """

    character_id: str
    match_id: str
    current_turn: int
    board_summary: BoardSummary
    mood: MoodState
    last_player_uci: str | None
    last_player_chat: str | None
    last_moves_san: list[str] = field(default_factory=list)
    recent_chat: list[str] = field(default_factory=list)  # last ~5 messages for context
    opening_label: str | None = None
    current_player_id: str | None = None
    opponent_style_features: dict | None = None


# --- Pipeline --------------------------------------------------------------


def _context_tokens(inp: SubconsciousInput) -> set[str]:
    themes: list[str] = []
    themes.extend(inp.board_summary.pinned_pieces)
    themes.extend(inp.board_summary.hanging_pieces)
    themes.extend(inp.board_summary.king_safety_concerns)
    themes.append(inp.board_summary.phase)
    themes.extend(inp.last_moves_san)
    return build_context_tokens(
        board_prose=inp.board_summary.prose,
        opening_label=inp.opening_label,
        opponent_style_features=inp.opponent_style_features,
        last_player_chat=inp.last_player_chat,
        tactical_themes=themes,
    )


def _query_text(inp: SubconsciousInput) -> str:
    parts = [inp.board_summary.prose]
    if inp.last_player_chat:
        parts.append(inp.last_player_chat)
    if inp.opening_label:
        parts.append(inp.opening_label)
    return " ".join(parts)


def _score_candidate(
    memory: Memory,
    *,
    semantic_score: float,
    context_tokens: set[str],
    mood: MoodState,
    current_player_id: str | None,
    opponent_style_features: dict | None,
    now: datetime,
) -> ScoreBreakdown:
    trigger = trigger_match_score(memory, context_tokens)
    opponent = opponent_relevance_score(
        memory,
        current_player_id=current_player_id,
        opponent_style_features=opponent_style_features,
    )
    mood_align = mood_alignment_score(memory, mood)
    recency = recency_penalty(memory, now=now)
    total = aggregate_scores(
        semantic=semantic_score,
        trigger=trigger,
        opponent=opponent,
        mood=mood_align,
        recency=recency,
    )
    return ScoreBreakdown(
        memory_id=memory.id,
        semantic=semantic_score,
        trigger=trigger,
        opponent=opponent,
        mood=mood_align,
        recency=recency,
        total=total,
    )


def _auto_reason(breakdown: ScoreBreakdown, memory: Memory) -> str:
    axis = breakdown.dominant_axis()
    if axis == "semantic":
        return "board + chat context resonates with this memory"
    if axis == "trigger":
        return "direct trigger match with the current position"
    if axis == "opponent":
        return "this opponent (or their archetype) matches the memory"
    return "mood aligns with the memory's tone"


# --- LLM re-rank ----------------------------------------------------------


class _ReRankedMemory(BaseModel):
    """Shape the Flash Lite re-rank returns per item."""

    memory_id: str = Field(..., description="Must be one of the memory_ids shown in the prompt.")
    retrieval_reason: str = Field(
        ..., description="One short sentence explaining why this memory fits the current moment."
    )


_RERANK_ADAPTER = TypeAdapter(list[_ReRankedMemory])


def _rerank_prompt(
    *,
    character: Character,
    inp: SubconsciousInput,
    candidates: list[tuple[Memory, ScoreBreakdown]],
) -> str:
    lines: list[str] = []
    lines.append(
        f"You are the subconscious of {character.name}, a chess-playing character. "
        f"You receive {len(candidates)} candidate memories pre-ranked by a deterministic "
        f"scorer, and your job is to pick the {TOP_K_FINAL} that fit THIS MOMENT best. "
        f"Optimize for emotional resonance and narrative fit, not rote topical overlap."
    )
    lines.append("")
    lines.append(f"CURRENT BOARD SUMMARY:\n{inp.board_summary.prose}")
    lines.append("")
    lines.append(f"MOOD: polarity={mood_polarity_bucket(inp.mood)} ({inp.mood.to_dict()})")
    if inp.last_player_chat:
        lines.append(f"PLAYER JUST SAID: {inp.last_player_chat!r}")
    if inp.opponent_style_features:
        lines.append(f"OPPONENT FEATURES: {inp.opponent_style_features}")
    if inp.last_moves_san:
        lines.append("RECENT MOVES: " + ", ".join(inp.last_moves_san[-6:]))
    lines.append("")
    lines.append("CANDIDATES:")
    for i, (mem, breakdown) in enumerate(candidates, start=1):
        lines.append(
            f"[{i}] id={mem.id} (scorer={breakdown.total:.3f}, "
            f"scope={mem.scope.value}, valence={mem.emotional_valence:+.2f}, "
            f"triggers={mem.triggers})"
        )
        lines.append(f"    {mem.narrative_text}")
    lines.append("")
    lines.append(
        f"Pick the {TOP_K_FINAL} best memories for this moment. Return a JSON array of "
        f"{{memory_id, retrieval_reason}}. The memory_id MUST be one of the ids shown "
        f"above. The retrieval_reason should be one short sentence."
    )
    return "\n".join(lines)


def _run_llm_rerank(
    *,
    character: Character,
    inp: SubconsciousInput,
    candidates: list[tuple[Memory, ScoreBreakdown]],
    llm: LLMClient,
) -> list[_ReRankedMemory]:
    prompt = _rerank_prompt(character=character, inp=inp, candidates=candidates)
    try:
        result = llm.generate_structured(
            prompt=prompt,
            response_schema=list[_ReRankedMemory],
            response_adapter=_RERANK_ADAPTER,
            temperature=0.4,
            max_output_tokens=1024,
            call_tag=f"subconscious_rerank:{inp.match_id}",
        )
    except LLMError as exc:
        logger.warning("Subconscious LLM re-rank failed; falling back to scorer order: %s", exc)
        return []
    valid_ids = {m.id for m, _ in candidates}
    cleaned: list[_ReRankedMemory] = []
    seen: set[str] = set()
    for item in result:
        if item.memory_id in valid_ids and item.memory_id not in seen:
            cleaned.append(item)
            seen.add(item.memory_id)
    return cleaned


# --- Surface-count bookkeeping --------------------------------------------


def _record_surfacing(session: Session, memory_ids: list[str]) -> None:
    rows = get_by_ids(session, memory_ids)
    now = datetime.utcnow()
    for m in rows:
        m.surface_count = (m.surface_count or 0) + 1
        m.last_surfaced_at = now
    session.flush()


# --- Main entry point ------------------------------------------------------


def run_subconscious(
    session: Session,
    character: Character,
    inp: SubconsciousInput,
    *,
    llm: LLMClient | None = None,
) -> list[SurfacedMemory]:
    """Return up to `TOP_K_FINAL` surfaced memories for the Soul.

    Respects the 3-turn cache. Increments surface_count only for
    freshly-surfaced memories (cached returns don't re-increment).
    """
    cache_key = build_cache_key(
        last_player_uci=inp.last_player_uci,
        last_player_chat=inp.last_player_chat,
        mood=inp.mood,
    )
    cached = _read_cache(inp.match_id, cache_key, inp.current_turn)
    if cached is not None:
        logger.debug(
            "Subconscious cache hit (match=%s turn=%d key=%s)",
            inp.match_id,
            inp.current_turn,
            cache_key,
        )
        return cached

    # Fast-path: if this character has no memories, nothing to surface.
    from sqlalchemy import select as _select
    from app.models.memory import Memory as _Memory
    has_any = session.execute(
        _select(_Memory.id).where(_Memory.character_id == inp.character_id).limit(1)
    ).scalar_one_or_none()
    if not has_any:
        return []

    # Step 1: semantic pre-filter via vector search.
    query = _query_text(inp)
    try:
        query_vec = embed_text(query)
    except Exception as exc:
        logger.warning("Subconscious embedding failed, skipping semantic filter: %s", exc)
        query_vec = []

    if query_vec:
        hits = vector_store.search(
            session,
            query_embedding=query_vec,
            k=CANDIDATE_POOL,
            character_id=inp.character_id,
        )
        semantic_by_id: dict[str, float] = {h.memory_id: h.score for h in hits}
        candidate_ids = list(semantic_by_id.keys())
    else:
        # No vector — fall back to all memories for this character.
        from sqlalchemy import select
        from app.models.memory import Memory as _Mem
        rows = list(
            session.execute(select(_Mem).where(_Mem.character_id == inp.character_id)).scalars()
        )
        candidate_ids = [r.id for r in rows]
        semantic_by_id = {cid: 0.0 for cid in candidate_ids}

    if not candidate_ids:
        logger.info("Subconscious: no memories available for character %s", inp.character_id)
        return []

    candidates_mem = get_by_ids(session, candidate_ids)
    # Preserve vector-search order.
    by_id = {m.id: m for m in candidates_mem}
    ordered_mems = [by_id[cid] for cid in candidate_ids if cid in by_id]

    # Step 2: weighted multi-axis scoring.
    ctx_tokens = _context_tokens(inp)
    now = datetime.utcnow()
    scored: list[tuple[Memory, ScoreBreakdown]] = []
    for mem in ordered_mems:
        scored.append(
            (
                mem,
                _score_candidate(
                    mem,
                    semantic_score=semantic_by_id.get(mem.id, 0.0),
                    context_tokens=ctx_tokens,
                    mood=inp.mood,
                    current_player_id=inp.current_player_id,
                    opponent_style_features=inp.opponent_style_features,
                    now=now,
                ),
            )
        )
    scored.sort(key=lambda pair: pair[1].total, reverse=True)

    # Step 3: decide re-rank vs direct.
    top_8 = scored[:TOP_K_FOR_RERANK]
    reasons: dict[str, str] = {}
    use_llm = False
    if len(top_8) >= 2:
        margin = top_8[0][1].total - top_8[1][1].total
        use_llm = margin <= LLM_RERANK_MARGIN

    final_order: list[tuple[Memory, ScoreBreakdown]]
    if use_llm:
        llm_client = llm
        if llm_client is None:
            try:
                llm_client = get_llm_client()
            except LLMError as exc:
                logger.warning("Subconscious: no LLM available, skipping re-rank: %s", exc)
                llm_client = None
        rerank_items: list[_ReRankedMemory] = []
        if llm_client is not None:
            rerank_items = _run_llm_rerank(
                character=character,
                inp=inp,
                candidates=top_8,
                llm=llm_client,
            )
        if rerank_items:
            id_to_pair = {m.id: (m, b) for m, b in top_8}
            final_order = [id_to_pair[r.memory_id] for r in rerank_items[:TOP_K_FINAL] if r.memory_id in id_to_pair]
            reasons = {r.memory_id: r.retrieval_reason for r in rerank_items}
        else:
            # LLM failed or disabled — use deterministic order.
            final_order = scored[:TOP_K_FINAL]
    else:
        final_order = scored[:TOP_K_FINAL]

    # Step 4: assemble output.
    surfaced: list[SurfacedMemory] = []
    for mem, breakdown in final_order:
        reason = reasons.get(mem.id) or _auto_reason(breakdown, mem)
        surfaced.append(
            SurfacedMemory(
                memory_id=mem.id,
                narrative_text=mem.narrative_text,
                triggers=list(mem.triggers or []),
                relevance_tags=list(mem.relevance_tags or []),
                emotional_valence=float(mem.emotional_valence),
                scope=mem.scope.value if hasattr(mem.scope, "value") else str(mem.scope),
                score=breakdown.total,
                retrieval_reason=reason,
                from_cache=False,
            )
        )

    # Step 5: bookkeeping + cache.
    if surfaced:
        _record_surfacing(session, [s.memory_id for s in surfaced])
    _write_cache(inp.match_id, cache_key, inp.current_turn, surfaced)

    return surfaced
