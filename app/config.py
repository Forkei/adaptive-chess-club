from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.1-flash-lite-preview"

    database_url: str = "sqlite:///./metropolis_chess.db"

    log_dir: str = "logs"
    log_level: str = "INFO"

    memory_gen_target: int = 40
    memory_gen_min: int = 30
    memory_gen_max: int = 50

    # Phase 2 — engine + infra
    redis_url: str = ""
    maia2_cache_dir: str = ""
    stockfish_path: str = ""

    # Phase 3b — Socket.IO real-time
    # How long a player has to reconnect after dropping their socket before the
    # match is abandoned. Short values in tests, 300s in prod.
    match_disconnect_cooldown_seconds: int = 300
    # Minimum interval between two `player_chat` events on the same socket.
    # Excess are rejected with a `player_chat_rate_limited` event, not a disconnect.
    player_chat_min_interval_ms: int = 500
    # Pending-chat FIFO cap merged into the next Subconscious call.
    pending_chat_max_messages: int = 10
    pending_chat_max_chars: int = 2000
    # Fixed UI-latency budget added to the engine's time budget to produce the
    # `agent_thinking.eta_seconds` hint — accounts for Soul + network.
    agent_thinking_soul_overhead_seconds: float = 1.5

    @property
    def log_path(self) -> Path:
        p = Path(self.log_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
