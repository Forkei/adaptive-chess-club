"""Subconscious + Soul — the two LLM-backed agents called per character turn.

The Subconscious decides which memories are top-of-mind right now.
The Soul decides what (if anything) the character says and how mood
updates. Both run every character turn (Soul unconditionally — silence
is a valid output).

See `app/schemas/agents.py` for the structured-output shapes.
"""
