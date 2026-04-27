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

    # Patch Pass 1 — periodic housekeeping.
    # Matches with status=in_progress + move_count < threshold + started_at older
    # than this many hours are reaped to ABANDONED by the startup/periodic sweep.
    stale_match_reaper_hours: int = 1
    stale_match_move_threshold: int = 2
    # MatchAnalysis rows in RUNNING older than this many minutes are marked FAILED
    # on startup + periodic sweep — avoids zombie rows after a server crash.
    stuck_analysis_minutes: int = 10
    # Periodic sweep cadence (both reapers run in one coroutine).
    housekeeping_interval_seconds: int = 300

    # Patch Pass 2 Item 5 — chat-triggered immediate Soul response.
    # When a player chat arrives and no character turn is in flight, fire a
    # lightweight Soul call at most once every this many milliseconds per match.
    # Prevents spam-chat cost blow-ups. Excess messages batch for the next
    # eligible response.
    chat_triggered_soul_min_interval_ms: int = 10000

    # Phase 4.0a — email/password auth.
    # Session cookie is unsigned (value is the player UUID). Kept for forward
    # compatibility if we later want to sign it with HMAC.
    session_secret: str = ""
    password_min_length: int = 8
    password_reset_token_ttl_minutes: int = 60
    # SMTP disabled by default → password-reset links are written to the log
    # dir (dev mode). Fill these in .env for real mail.
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from: str = "no-reply@metropolis-chess.local"

    # Single-character mode: create/edit/clone character API endpoints are
    # disabled unless this env var is set. Tests flip it to True in their scope.
    allow_character_api: bool = False

    # Username (case-insensitive) allowed to access /admin/feedback.
    # Defaults to the project owner; set to "" to disable the admin view.
    admin_username: str = "forkei"

    # Block 17: $CLAY simulated economy.
    # Starting grant in cents (default 10000 = 100 $CLAY).
    starting_clay_grant: int = 10000
    # Max stake per single agent-vs-character match (cents). 10000 = 100 $CLAY.
    max_stake_cents: int = 10000

    @property
    def log_path(self) -> Path:
        p = Path(self.log_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
