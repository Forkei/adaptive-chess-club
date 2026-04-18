"""Single entry point for all LLM calls.

Wraps `google-genai` with:
- Pydantic structured output (list or model schemas)
- Exponential-backoff retry on transient errors
- JSONL call logging

All LLM calls in the app go through this module.
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Any, TypeVar

from pydantic import BaseModel, TypeAdapter

from app.config import get_settings
from app.llm.call_log import log_call

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Raised when an LLM call fails after retries or returns unusable output."""


class LLMClient:
    """Thin wrapper around google-genai's `Client`.

    Instantiate directly for tests (pass a fake `_client`), or obtain the
    app-wide singleton via `get_llm_client()`.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        max_retries: int = 3,
        initial_backoff_s: float = 1.0,
        _client: Any = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.initial_backoff_s = initial_backoff_s

        if _client is not None:
            self._client = _client
        else:
            from google import genai  # lazy import so tests can stub it

            self._client = genai.Client(api_key=api_key)

    def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: Any,
        response_adapter: TypeAdapter[T] | None = None,
        temperature: float = 0.9,
        max_output_tokens: int | None = None,
        call_tag: str = "",
    ) -> Any:
        """Generate a response and parse it via `response_schema`.

        `response_schema` may be a Pydantic model class, `list[Model]`, or any
        type the google-genai SDK accepts for `response_schema`. If
        `response_adapter` is provided, it's used to re-validate whatever the
        SDK returned — useful when `response.parsed` is None (SDK fallback
        to raw text) or when the SDK's parse step is flakey.
        """
        from google.genai import types  # lazy import

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=response_schema,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            started = time.perf_counter()
            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )
                latency_ms = int((time.perf_counter() - started) * 1000)
                parsed = self._extract_parsed(response, response_adapter)

                log_call(
                    {
                        "tag": call_tag,
                        "model": self.model,
                        "prompt_chars": len(prompt),
                        "response_chars": len(getattr(response, "text", "") or ""),
                        "latency_ms": latency_ms,
                        "attempt": attempt,
                        "success": True,
                    }
                )
                return parsed
            except LLMError:
                raise
            except Exception as exc:
                latency_ms = int((time.perf_counter() - started) * 1000)
                last_exc = exc
                logger.warning(
                    "LLM call failed (attempt %d/%d, tag=%s): %s",
                    attempt,
                    self.max_retries,
                    call_tag,
                    exc,
                )
                log_call(
                    {
                        "tag": call_tag,
                        "model": self.model,
                        "prompt_chars": len(prompt),
                        "latency_ms": latency_ms,
                        "attempt": attempt,
                        "success": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                if attempt < self.max_retries:
                    time.sleep(self.initial_backoff_s * (2 ** (attempt - 1)))

        raise LLMError(f"LLM call failed after {self.max_retries} attempts: {last_exc}")

    @staticmethod
    def _extract_parsed(response: Any, adapter: TypeAdapter[Any] | None) -> Any:
        parsed = getattr(response, "parsed", None)
        if parsed is not None:
            return parsed
        text = getattr(response, "text", None)
        if text and adapter is not None:
            try:
                return adapter.validate_json(text)
            except Exception as exc:
                raise LLMError(f"Failed to parse LLM JSON output: {exc}") from exc
        raise LLMError("LLM returned no parsed output and no adapter was provided")


@lru_cache(maxsize=1)
def get_llm_client() -> LLMClient:
    settings = get_settings()
    if not settings.gemini_api_key:
        raise LLMError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and fill it in."
        )
    return LLMClient(api_key=settings.gemini_api_key, model=settings.gemini_model)
