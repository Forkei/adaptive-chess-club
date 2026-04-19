"""Post-match processing pipeline.

Runs after a match finalizes (completion, resign, or future
disconnect-as-resign). Five steps, each in its own try/except so a
failure doesn't kill the rest:

1. Engine analysis — Stockfish replay, eval_before/after/loss per move.
2. Feature extraction — style features for the OpponentProfile.
3. Elo ratchet — apply spec formula to character_elo + floor_elo.
4. Memory generation — LLM produces 1-3 memories capturing this match.
5. Narrative summary — LLM updates OpponentProfile.narrative_summary.

The orchestrator (`processor.py`) runs in a daemon thread. Status +
step progress is persisted on MatchAnalysis so the client can poll.
"""
