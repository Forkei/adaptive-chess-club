"""Sentence-transformer embeddings for memory retrieval.

Single model instance, lazy-loaded, reused for all embedding calls. The
default is `sentence-transformers/all-MiniLM-L6-v2` (384 dims, CPU-fast,
~90MB download on first use).

Tests monkey-patch `_get_model` to avoid downloading the real model.
"""

from __future__ import annotations

import logging
import threading
from typing import Iterable, Protocol

from app.models.memory import Memory

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


class _EncoderLike(Protocol):
    def encode(self, texts: list[str], **kwargs) -> object: ...


_MODEL: _EncoderLike | None = None
_MODEL_LOCK = threading.Lock()


def _get_model() -> _EncoderLike:
    """Lazy singleton for the sentence-transformers model."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Run: pip install sentence-transformers"
            ) from exc
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
        _MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _MODEL


def embed_texts(texts: Iterable[str]) -> list[list[float]]:
    """Batch-encode `texts`. Returns a list of same-length float lists.

    Empty strings are allowed but produce low-quality embeddings; the
    caller should build meaningful input via `build_memory_embedding_input`.
    """
    text_list = list(texts)
    if not text_list:
        return []
    model = _get_model()
    vectors = model.encode(text_list, convert_to_numpy=True, show_progress_bar=False)
    return [list(map(float, v)) for v in vectors]


def embed_text(text: str) -> list[float]:
    """Single-text wrapper around `embed_texts`."""
    result = embed_texts([text])
    return result[0]


def build_memory_embedding_input(
    *,
    narrative_text: str,
    triggers: list[str],
    relevance_tags: list[str],
) -> str:
    """Assemble the string the embedder sees for a memory.

    Kept as a helper so creation and backfill stay in sync.
    """
    trig = " ".join(triggers) if triggers else ""
    tags = " ".join(relevance_tags) if relevance_tags else ""
    return f"{narrative_text} | {trig} | {tags}".strip()


def embedding_input_for(memory: Memory) -> str:
    return build_memory_embedding_input(
        narrative_text=memory.narrative_text,
        triggers=list(memory.triggers or []),
        relevance_tags=list(memory.relevance_tags or []),
    )
