# Phase 4.3 — Agent evolution

Characters evolve after each match. Same character for everyone (global
drift), bounded per match, clamped to identity. Elo is already handled
by Phase 4.0b; this phase handles the human-like learning layer on top.

## Scope

Five channels, in priority order:

1. **Stylistic drift** — `aggression`, `risk_tolerance`, `patience`,
   `trash_talk` sliders shift a small amount based on match outcome +
   opponent quality.
2. **Opening-book evolution** — per-opening score updated by a simple
   EMA of (opening_played · result_signal). Over time a character favours
   openings that won and drifts away from openings that lost.
3. **Trap memory** — pattern-specific counters ("fell for scholar's mate
   vs. @alice"). Probability of falling again drops but never zeros —
   humans still slip.
4. **Tone drift** — confidence/tilt baselines shift over streaks. Not a
   slider; a bias the Soul sees in its mood state.
5. **Hard clamps** — a Viktor who loses 50 games in a row can't turn
   into a pacifist. Sliders cannot drift more than ±2 from base;
   opening-book weights cannot flip sign on their basis openings.

## Invariants

- **Private matches skip the whole evolution step** (per Phase 4.0b
  memory). Set once: `if match.is_private: return`.
- **Global, not per-player.** One evolution state row per character.
  Per-player memory (who bullied whom) stays in `OpponentProfile` and
  character memories.
- **Per-match deltas are small.** A single match can only move a slider
  by ≤ 1 step, a tone baseline by ≤ 0.05, an opening EMA by ≤ 0.1.
- **Identity clamps bound cumulative drift.** Character's base sliders
  are the anchor. Cumulative drift is capped at ±2 slider steps from
  base. The character can change — slowly — but can't become unrecognisable.
- **Cadence: every match.** Kept batch-friendly so we can flip to
  "every N matches" later without rewriting the pipeline.

## Data model

`character_evolution_state` — one row per character, lazy-created on
first post-match run.

| column | type | notes |
|---|---|---|
| `character_id` (PK) | String(36) FK | one row per character |
| `slider_drift` | JSON | `{aggression, risk_tolerance, patience, trash_talk}` — signed deltas from base; bounded [-2, +2] |
| `opening_scores` | JSON | `{"<opening_label>": float}` — EMA-updated result signal in [-1, +1] |
| `trap_memory` | JSON | list of `{"pattern": str, "fell_for": int, "avoided": int, "last_seen_at": iso}` |
| `tone_drift` | JSON | `{"confidence_baseline": float, "tilt_baseline": float}` in [-0.3, +0.3] |
| `matches_processed` | int | counter |
| `last_updated_at` | datetime | |
| `last_match_id` | str nullable | for idempotency check — skip if we've already processed this match |

Migration `0012_character_evolution_state`.

## Drift math

Let `win_value`: `+1` on a win (character POV), `-1` on loss, `0` on draw.
Let `opponent_strength`: `player_elo_at_start / character_elo_at_start` (centered
around 1.0). Strong opponents amplify drift; weak opponents attenuate it.

```
signal        = win_value * clamp(opponent_strength, 0.3, 2.0)
```

**Slider drift.** Each match picks at most ONE slider to nudge,
selected by which one would have helped most given the outcome:

- Lost, played cautiously (avg ACPL < 30) → nudge `aggression` +0.5.
- Lost, played recklessly (avg ACPL > 80) → nudge `patience` +0.5.
- Won, opponent higher-rated → nudge `confidence` in tone (not a slider).
- On tilt (recent loss streak ≥ 3) → nudge `trash_talk` slightly up or
  down toward the character's base depending on direction-of-drift.

```
new_drift = clamp(old_drift + delta, -2.0, +2.0)
```

Cap: at most one slider updated per match. Delta magnitude ≤ 0.5.

**Opening-book EMA.**
For each opening the character played in the match:
```
score'       = 0.9 * score_prev  +  0.1 * signal
```
Only applied to openings in `character.opening_preferences` or to
openings whose score has been built up over ≥ 3 matches.

**Trap memory.**
A "trap" is a sharp loss in ≤ 15 plies, or a loss to a named mating
pattern detectable by the analysis step. Detection is cheap: reuse
`MatchAnalysis.critical_moments` — the first critical moment in the
first 12 plies with swing ≥ 400cp is tagged as a trap and fed into
`trap_memory`.

```
if detected_trap(pattern):
    entry = trap_memory[pattern] or new(entry)
    if character_fell: entry.fell_for += 1
    else:              entry.avoided += 1
    entry.last_seen_at = now
```

A learning Memory row is generated the FIRST time a pattern is seen
(narrative: "Last time an opponent tried X, I stepped right into it
..."). Subsequent hits only bump the counter — no new memory row
per match (prevents the Subconscious from drowning in redundant copies).

**Tone drift.** EMA toward current streak:
```
confidence_baseline'  = 0.95 * confidence_prev + 0.05 * clamp(win_streak / 5, -0.3, +0.3)
tilt_baseline'        = 0.95 * tilt_prev       + 0.05 * clamp(-loss_streak / 5, -0.3, +0.3)
```

Applied every match. Fed to the Soul as an additive nudge on the
initial `MoodState` before each turn.

## Pipeline

Added as a new step after `elo_ratchet`, before `memory_generation` in
`app/post_match/processor.py`. That way generated memories can reference
newly-detected traps.

```
engine_analysis → feature_extraction → elo_ratchet
  → evolution                ← NEW
  → memory_generation → narrative_summary
```

Contract: `apply_evolution(session, match, analysis)` — raises on
programmer errors, swallows LLM errors (shouldn't invoke LLM anyway —
the hook is pure data). Idempotent via `last_match_id` guard.

## Integration with Subconscious

- Learned memories from trap memory use a new `MemoryType.LEARNING`
  scope. They enter the normal vector-retrieval flow; the Subconscious
  doesn't need changes.
- Tone drift is read at Soul call time — `_build_soul_input`
  adds `tone_drift` to the `mood` parameter before passing to the Soul.
  No Soul-side changes needed.
- Slider drift is read whenever the Director builds the engine config
  and whenever the style prompt is rendered. Effective slider =
  `character.slider + drift`.

## Private-match exclusion

At entry: `if match.is_private: return` — no state read, no state
write, no memory generation of learned entries. Logged at INFO so a
demo can grep evolution.log to verify the guardrail is working.

## Testing

- **Unit tests** per math function (slider delta selection, opening
  EMA step, trap detection, tone EMA, clamps).
- **Integration test** — simulate a 50-match burst against a fixed
  opponent profile and assert cumulative drift stays in bounds.
- **Private-match test** — confirm `apply_evolution` is a no-op on
  `match.is_private = True`.
- **Idempotency test** — re-running the pipeline on the same match
  doesn't double-apply drift.

## What this phase does NOT do

- Does not build a multi-player personality model (global only).
- Does not feed PvP matches into evolution (only PvE — characters are
  PvE only; PvP is player-vs-player).
- Does not build a UI for viewing a character's drift. Phase 4.5 polish
  pass adds a read-only "this character has learned…" panel on the
  detail page if there's time.
