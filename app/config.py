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

    @property
    def log_path(self) -> Path:
        p = Path(self.log_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
