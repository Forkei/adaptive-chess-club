---
title: Metropolis Chess Club
emoji: ♞
colorFrom: indigo
colorTo: amber
sdk: docker
app_port: 7860
pinned: false
---

# Metropolis Chess Club

A chess-playing AI character platform. Users create or pick an AI opponent — each with its own backstory, personality sliders, opening preferences, and lived-in memories — then play against it with real-time chat.

The runtime has three layers:

- **Soul** — an LLM that handles chat, personality, and intent (**Phase 2b**)
- **Director** — deterministic code that translates the Soul's mood/intent into engine config (**Phase 2a, built**)
- **Body** — a chess engine: Maia-2 for human-like play, Stockfish for high-skill "beast mode" (**Phase 2a, built**)

A **Subconscious** agent retrieves relevant memories every turn and feeds them to the Soul (**Phase 2b**).

## Status

| Phase | Scope | State |
|-------|-------|-------|
| **1**   | Character data model, backstory → memories generator, presets, REST/HTML UI | ✅ shipped |
| **2a**  | Engines + Director + mood + Redis + match lifecycle + playable board (character is silent) | ✅ shipped |
| **2b-1**| Embeddings + Subconscious (multi-axis retrieval, 3-move cache) + Soul (structured chat/mood/opponent notes) wired into `/move` | ✅ shipped |
| **2b-2**| Post-match processor (engine analysis, feature extraction, Elo ratchet application, memory generation, narrative summary, background threading, status polling) | ✅ shipped |
| **2b-3**| UI polish (chat log, memory ribbon, emotion indicator, post-match summary) + live end-to-end test | ⏳ next |
| **3a**  | Username login, character ownership, content rating, Alembic migrations | ✅ shipped |
| **3b**  | Socket.IO real-time gameplay, chat-during-thinking, disconnect handling, streamed post-match | ✅ shipped |
| **3c**  | Match discovery, spectating + crowd-noise chat, leaderboards + hall of fame, frontend polish | ✅ shipped |

### What 2a adds

- **Engine layer**: `ChessEngine` ABC + `MoveResult` + `EngineConfig`. Three implementations:
  - **Maia-2** (human-like, rapid model, 1100–1900 Elo band, CPU)
  - **Stockfish** (via UCI, `UCI_LimitStrength` + `UCI_Elo`, top-3 moves with centipawn evals)
  - **MockEngine** (deterministic first-legal-move, always available, used in tests and dev-without-engines)
- **Director** (`app/director/director.py`): pure function from `(character, mood, opponent_profile, match_context)` to `EngineConfig`. Implements:
  - **Effective Elo** = `current_elo + 100·confidence − 150·tilt`, clamped `[floor_elo, max_elo]`
  - **Engine selection**: Stockfish beast-mode when `effective_elo > 2100` or `aggression ≥ 9 ∧ confidence > 0.7`. Stockfish fallback below Maia's range or when Maia-2 isn't installed.
  - **Time budget** keyed off `patience` (1→0.5s, 10→5s), discounted by mood aggression, capped at 8s.
- **Mood** (`app/director/mood.py`): 4-axis state (`aggression / confidence / tilt / engagement`) initialized from character sliders (`aggression → mood.aggression`, `trash_talk` biases `engagement`). Exponential smoothing, τ = 3 moves. Persisted in Redis keyed by match.
- **Elo ratchet** (`app/director/elo.py`): `±10%` of signed outcome, clamped `±30` / match; floor rises `+25` after 3 consecutive currents `> floor + 100`. Non-adaptive characters don't move at all.
- **Redis wrapper** with in-process dict fallback (logs a warning).
- **Board abstraction** (`app/engine/board_abstraction.py`): turns a `chess.Board` into a `BoardSummary` (material, castling rights, central-pawn structure, pinned/hanging pieces, king-safety flags, phase, optional eval prose). The Soul in Phase 2b consumes this instead of a FEN — the whole point is to prevent LLM chess hallucinations.
- **Match models**: `Player`, `Match`, `Move`, and `OpponentProfile` (schema only in 2a; populated in 2b).
- **Match REST API**:
  - `POST /api/matches` (cookie-authed Guest player auto-created)
  - `GET /api/matches/{id}`
  - `POST /api/matches/{id}/move` (JSON `{uci}`)
  - `POST /api/matches/{id}/resign`
  - `GET /api/matches/{id}/moves`
  - `POST /api/players` / `GET /api/players/me`
- **Playable HTML UI** at `/matches/{id}` — `chess.js` + `chessboard.js` via CDN, board + move list, resign button. No chat field in 2a.
- **Character state gets 3 new Elo fields**: `current_elo`, `floor_elo`, `max_elo`. **Phase 1 DBs are incompatible** — delete the SQLite file when upgrading.

### What 2b Slice 1 adds

- **Embedding pipeline** (`app/memory/embeddings.py`): single `sentence-transformers/all-MiniLM-L6-v2` instance, lazy-loaded, reused for all memory embeddings (384 dims). Memories are embedded inline at creation in `bulk_create`; the backfill script (`scripts/backfill_embeddings.py`) covers pre-2b rows.
- **Vector store** (`app/memory/vector_store.py`): thin interface over an `embedding` JSON column on `Memory`. Cosine similarity computed in-Python with numpy. For a corpus of ~40-50 memories per character × a handful of characters, in-memory scoring is sub-millisecond; the interface (`upsert`, `search`, `get_embedding`) is stable so Phase 3 can swap in sqlite-vec without touching callers. On a pre-existing SQLite DB the `memories.embedding` column is added idempotently at startup (`ensure_embedding_column`).
- **Subconscious** (`app/agents/subconscious.py`): runs before the Soul on every character turn. Multi-axis scoring — **semantic** (0.35) · **trigger** (0.25) · **opponent** (0.15) · **mood alignment** (0.10) · **recency penalty** (-0.15). If top-1 beats top-2 by `> 0.15`, return top-5 directly; otherwise send top-8 to Gemini Flash Lite for structured re-rank with one-sentence `retrieval_reason` per memory. Per-match cache with TTL = 3 character turns keyed by `(last_player_uci, chat_hash, mood_polarity_bucket)` — cached returns carry `from_cache=True` and do NOT re-increment `surface_count`.
- **Soul** (`app/agents/soul.py`): runs after the engine move on every character turn. Structured output (`SoulResponse`) containing `speak` (nullable — silence is the default), `emotion` + `emotion_intensity`, `mood_deltas` (±0.1 per axis), `note_about_opponent` (queued for post-match), `referenced_memory_ids` (sanitized to be a subset of the surfaced set), and an optional `internal_thinking` debug field. Voice rules, mood distortion, and speaking discipline live in the system prompt; per-turn dynamic context lives in the user prompt. LLM failures silently degrade to a neutral silent response so gameplay never stalls.
- **Mood polarity buckets** (`app/agents/retrieval.py`): `deflated / tense / guarded / steady / confident / dominant`, derived from a weighted composite of confidence, aggression, tilt, and engagement. Used for Subconscious cache keying and available to downstream UI snippets.
- **Match service integration**: `apply_player_move` now accepts `player_chat`, runs Subconscious → Soul after the engine move, applies `mood_deltas` to raw mood and re-smooths, and persists `agent_chat_after` / `surfaced_memory_ids` on the Move row. `note_about_opponent` is queued into `match.extra_state['pending_opponent_notes']` for Slice 2. The HTTP response adds an `agent_turn` field with `speak`, `emotion`, `emotion_intensity`, and surfaced memory snippets for the UI.
- **`MoveSubmit.chat`**: optional string (≤500 chars) the player attaches to their move; stored on `Move.player_chat_before` and visible to the Subconscious next turn.

### What 2b Slice 2 adds

The post-match processor runs the moment a match finalizes (natural end or resign) in a daemon thread, so the HTTP response for the final move returns immediately. A new row in `match_analyses` tracks per-match state; clients poll `GET /api/matches/{id}/post_match_status` every ~2 s to get `status` (`pending / running / completed / failed`), `steps_completed`, error messages, and final outputs (Elo delta, critical moments, generated memories, updated narrative summary).

Pipeline (5 steps, each in its own try/except — partial success leaves later steps degraded but not broken):

1. **Engine analysis** (`app/post_match/analysis.py`): Stockfish replays every position at 0.3 s/move, recording `eval_before / eval_after / best_move / eval_loss` and flagging blunders (eval_loss ≥ 200 cp). When Stockfish is unavailable, this step records `status=skipped` and downstream steps proceed with lighter signals. `identify_critical_moments` picks the top ~6 moves by a `loss × 2 + swing − 150` score — blunders and sign-flips dominate.
2. **Feature extraction** (`app/post_match/features.py`): computes player-side `aggression_index` (captures/checks/promotions), `typical_opening_eco/name/group` via the new `classify_opening(san_moves)` helper, `blunder_rate` normalized per 40 moves, `time_trouble_blunders` in the last 10 half-moves, `preferred_trades` per piece type, and `phase_strengths` (mean eval loss per opening/middlegame/endgame). `merge_features(previous, new, prior_games)` runs a weighted running average so multi-match profiles drift rather than snap.
3. **Elo ratchet** (`app/post_match/elo_apply.py`): `outcome_delta = win(+200) / draw(0) / loss(-200)`; `move_quality_delta = clamp((opponent_total − character_total)/10, ±100)` over summed per-move blunder magnitudes clamped at 300 cp; raw halved when `match.move_count < 10`; abandoned matches use outcome only. Applied via `apply_elo_ratchet` from Phase 2a — 10 % gain, ±30 per match clamp, floor ratchet `+25` when the last three matches sit ≥ 100 above the floor.
4. **Memory generation** (`app/post_match/memory_gen.py`): LLM (`google-genai` structured output, `list[_MatchMemory]`) receives character voice + match outcome + critical moments + opponent features (before + after) + drained `pending_opponent_notes` + a few-shot of existing memories. Asks for 1–3 memories scoped `MATCH_RECAP` / `OPPONENT_SPECIFIC`, validates that every memory has ≥ 3 triggers, retries once with explicit feedback if any memory falls short, then embeds and persists via `bulk_create(embed=True)`.
5. **Narrative summary** (`app/post_match/memory_gen.py`): second LLM call rewrites `OpponentProfile.narrative_summary` — first-person, ≤ 3 sentences, updates rather than replaces.

Rage-quit handling: `match.status == ABANDONED` skips move-quality and tells the memory generator to treat the match as rage-quit, letting the LLM's voice judge whether the character is bitter or amused.

### What 3a adds

- **Username login** (no password, no real auth — this is still a toy):
  - `GET /login` → username form; `POST /login` creates the account if new, reuses it if the username exists. Existing guest cookies can claim a real username (renames the row, preserves matches and memories).
  - `GET /logout` clears the cookie.
  - `GET /api/me` returns the current `Player` or 401.
  - Protected routes: HTML redirects to `/login`, API returns 401. Open: `/`, `/login`, `/logout`, `/api/me`, static assets.
  - Username rules: lowercase letters/digits/underscore, 3–24 chars, case-insensitive.
- **Character ownership**:
  - `Character.owner_id` (NULL = system-owned preset). Non-preset characters created pre-3a were reassigned on migration to a synthetic `legacy_system` Player (`display_name="Legacy"`).
  - `Visibility` enum (`public` / `private`). Private characters are 404 for non-owners.
  - Owner-only: `PATCH /api/characters/{id}`, `DELETE /api/characters/{id}`, `POST /api/characters/{id}/regenerate_memories`.
  - Anyone can `POST /api/characters/{id}/clone` — copies sheet, voice, quirks, sliders; clone gets fresh Elo state + fresh memory generation (independent of source's generation state). Memories are not copied.
  - Presets cannot be edited or deleted (403). Clone them first.
- **Content rating**:
  - Per-character `content_rating` enum: `family` / `mature` / `unrestricted`. Presets: Viktor + Kenji are `mature`; Margot + Archibald are `family`.
  - Per-player `max_content_rating`: characters rated above this are filtered out of listings; direct access returns 404 on `/api/` and a friendly "hidden by your content preference" page on `/characters/{id}`.
  - Rating is injected into every prose-producing LLM prompt: backstory memory generation, Soul system prompt, post-match memory generation, narrative summary. See `app/characters/content_rating_prompts.py`.
  - Player settings page (`/settings`) lets users edit `display_name` and `max_content_rating`. Username is immutable in 3a.

### What 3b adds

Real-time gameplay over Socket.IO, replacing the REST `POST /api/matches/{id}/move` round-trip as the
live transport. REST stays functional as a fallback and returns `X-Deprecated: Use Socket.IO /play namespace`
so anyone hitting the API directly sees the migration path.

- **Transport**: `python-socketio` `AsyncServer` mounted alongside FastAPI via `socketio.ASGIApp`
  (`app/sockets/server.py`). One namespace, `/play`. One room per match (`match:<match_id>`).
  Cookie-based auth — the existing `player_id` cookie from Phase 3a is parsed on the handshake;
  no cookie ⇒ connection refused. Same-origin only (`cors_allowed_origins=[]`).
- **Event contract**: all payloads modelled with Pydantic in `app/sockets/events.py`. Client →
  server: `make_move`, `player_chat`, `resign`, `ping`, `request_state`. Server → client:
  `match_state`, `player_move_applied`, `agent_thinking`, `memory_surfaced`, `agent_move`,
  `agent_chat`, `mood_update`, `match_ended`, `post_match_status`, `post_match_complete`,
  `match_paused`, `match_resumed`, `player_chat_echoed`, `player_chat_rate_limited`, `pong`, `error`.
- **Event ordering per character turn** (enforced by `app/matches/streaming.py`):
  `player_move_applied → agent_thinking → memory_surfaced → agent_move → agent_chat → mood_update`.
  `memory_surfaced` deliberately fires **before** `agent_move` — the Subconscious runs
  concurrently with the engine (on the post-player / pre-engine board), so the memory
  ribbon populates during the "thinking" state. This is the headline UX property of 3b.
  `agent_thinking.eta_seconds` is an approximate `time_budget + 1.5s` (rounded to 0.5s);
  document this as a soft estimate — if the Soul returns `speak=None`, the user-visible
  turn ends at `agent_move` rather than the non-existent `agent_chat`.
- **Player chat while character is thinking**: `player_chat` events are accepted at any time,
  buffered into `Match.extra_state['pending_player_chat']` (FIFO, capped at 10 messages / 2000
  chars via `PENDING_CHAT_MAX_MESSAGES` / `PENDING_CHAT_MAX_CHARS`), and merged into the **next**
  Subconscious call's `recent_chat` context. Rate-limited to one chat per 500ms per socket
  (`PLAYER_CHAT_MIN_INTERVAL_MS`); excess ⇒ `player_chat_rate_limited` event (no disconnect).
  Echoed back as `player_chat_echoed` so the UI can un-fade its optimistic bubble.
- **Disconnect handling**: closing the tab severs the socket, which stamps
  `match.extra_state.disconnect_started_at` and arms an `asyncio` task (registry in
  `app/sockets/disconnect.py`). If the player reconnects before
  `MATCH_DISCONNECT_COOLDOWN_SECONDS` elapses (default 300s), the task is cancelled,
  `match_resumed` fires, and gameplay continues. If the cooldown fires, the match is
  marked `ABANDONED` (character wins — same as resign) and the post-match processor kicks
  off. Character chat stays frozen during the pause — no "waiting" messages, no
  impatience behaviour (the design explicitly chose this).
- **Content rating mid-match**: changing `max_content_rating` during a match does **not**
  cut the socket — the match plays out; the character just stops appearing in the
  browse list. The Soul's rating injection was baked in at match creation, not
  re-evaluated per turn.
- **Post-match via sockets**: the Phase 2b processor now accepts a
  `status_callback: Callable[[str, dict], None]`. The Socket.IO layer supplies one
  (`app/sockets/processor_callback.py`) that emits `post_match_status` events per step
  and `post_match_complete` with the summary URL. Bridge across the sync-thread/async-loop
  boundary uses `asyncio.run_coroutine_threadsafe` against the loop captured at app startup.
  The legacy polling endpoint `GET /api/matches/{id}/post_match_status` remains as a
  fallback for clients that reconnect after post-match completion.
- **Single-worker assumption**: the disconnect registry and room state are in-process.
  Multi-worker deployment requires Redis pub/sub (Phase 4). A startup check warns (but
  does not fail) when `WEB_CONCURRENCY` or `UVICORN_WORKERS` > 1.
- **UI** (`app/web/templates/play.html`): Socket.IO client via CDN, a persistent chat
  input (not gated on turn order), connection-status pill, thinking spinner with eta,
  memory ribbon populating on `memory_surfaced`, emotion indicator on `agent_chat`,
  disconnect overlay with live countdown, inline post-match progress → auto-redirect
  to `/matches/<id>/summary` on `post_match_complete`.

Testing:

- `tests/test_sockets_unit.py` — disconnect registry, pending-chat caps, processor callback.
- `tests/test_sockets_integration.py` — live Socket.IO via `socketio.AsyncClient` against
  an in-process uvicorn; covers event ordering, disconnect+reconnect resume, disconnect
  timeout → abandoned, chat-during-thinking draining into the next Subconscious call,
  and rate-limiting. Opt-in live variant asserts the memory-before-move ordering with
  real Gemini + Maia-2.
- `docs/phase_3b_manual_smoke.md` — browser runbook for the M.6–M.8 smoke steps.

### What 3c adds

Social layer on top of the Phase 3b real-time core.

- **Discovery page** (`/discovery`): live matches + recently finished matches + the full character grid, all filtered through a single `visible_character_filter` (content rating + visibility). Your own in-progress match is hidden from "live" (you're already in it); your completed matches do appear under "recently finished". A prominent "Browse live matches →" CTA on `/` keeps discovery findable.
- **Spectating** (`/matches/{id}/watch`): read-only board + agent chat + memory ribbon + mood, plus a separate "crowd noise" spectator chat panel. Participants see spectator chat in their own panel (with local mute) and a live spectator count. Characters never see spectators — `spectator_chat` is a separate event that bypasses `Match.extra_state.pending_player_chat`. Verified in `test_phase_3c_spectators.py::test_spectator_chat_does_not_reach_subconscious`.
- **Role-gated Socket.IO handlers**: `_on_connect` tags the session `role=participant` or `role=spectator`; `make_move`, `resign`, `player_chat` reject spectators with typed error events; `spectator_chat` rejects participants. Participant's `player_chat` is also broadcast to spectators as `player_chat_broadcast` so they see the dialogue.
- **Leaderboards** (`/leaderboard/characters`, `/leaderboard/players`): win-rate ranking with `all | 30d | 7d` windows, minimum-5-matches threshold *within* the selected window, content-rating filter applied at the query level. **Abandoned matches count as character wins** (rage-quit-prone characters should reflect that in their record). Current user's row is highlighted on the player leaderboard.
- **Hall of fame** on every character detail page: top 10 players vs. that character by wins.
- **Player profile** at `/players/{username}`: recent matches + characters created, visibility-filtered.
- **Indexes**: Alembic 0003 adds `ix_matches_status` and `ix_matches_ended_at` for the discovery / leaderboard queries. Idempotent against pre-existing DBs (skips if `matches` table absent, as in the pre-3a migration fixture).
- **Shared Jinja partials** (`app/web/templates/_partials/`): `character_card.html`, `match_row.html`, and `_macros.html` (rating / visibility / result chips) — used from `/`, `/discovery`, player profile, and the polish pass.

### What's deliberately NOT in 2a

- No Subconscious, Soul, or post-match processor
- No chat field on moves, no surfaced memories in responses
- No `OpponentProfile` aggregation logic
- No Socket.IO / real-time streaming
- No scheduler / idle detection

## Setup

**Supported dev targets:** macOS, Linux, Windows via WSL2. The Dockerfile is the contract — that's where Stockfish, Maia-2, and all engine deps are known to work. Windows-native is not supported.

### Docker (recommended)

**On Windows (WSL2):** Clone the repo to WSL, or open the Windows folder from inside WSL with `cd /mnt/c/path/to/repo`. Then run all Docker commands from a WSL terminal. First build takes 10–20 minutes — Maia-2 weights download plus torch install.

```bash
cp .env.example .env     # add GEMINI_API_KEY if you want memories generated
docker compose up --build
```

Open http://localhost:8000. Presets seed on first boot (memories generate in background if `GEMINI_API_KEY` is set). Maia-2 weights download during the first build (~300 MB) and are cached in a Docker volume.

### Local venv (character pages + mock engine only)

Useful for Phase 1 character-editing work without pulling torch / Maia-2:

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate   |   Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env
uvicorn app.main:app --reload
```

Character pages work fully; matches run against the MockEngine (deterministic, uninteresting). To get real play locally, either install `.[engine]` (`pip install -e ".[dev,engine]"`) and a Stockfish binary, or use Docker.

## Running tests

```bash
pytest                                   # mocks only; no engines required
RUN_LIVE_LLM_TESTS=1 pytest -m live      # opt-in, needs GEMINI_API_KEY
```

## Layout

```
app/
├── main.py               # FastAPI entry, startup seeding
├── config.py             # pydantic-settings
├── db.py                 # SQLAlchemy 2.0 engine + session
├── logging_config.py
├── redis_client.py       # Redis wrapper + in-memory fallback
├── llm/
│   ├── client.py         # google-genai wrapper
│   └── call_log.py       # JSONL call log
├── engine/               # Phase 2a
│   ├── base.py           # ChessEngine ABC, EngineConfig, MoveResult
│   ├── registry.py       # get_engine(), availability probing
│   ├── maia2_engine.py
│   ├── stockfish_engine.py
│   ├── mock_engine.py
│   └── board_abstraction.py
├── director/             # Phase 2a — no LLM calls here
│   ├── director.py       # choose_engine_config
│   ├── mood.py           # MoodState + smoothing + persistence
│   └── elo.py            # Elo ratchet
├── matches/              # Phase 2a
│   └── service.py        # create / move / resign / turn loop
├── models/               # SQLAlchemy ORM (Character, Memory, Player, Match, Move, OpponentProfile)
├── schemas/              # Pydantic v2
├── characters/
│   ├── style.py, openings.py
│   ├── memory_generator.py, presets.py, seed.py
├── memory/crud.py
├── api/characters.py, players.py, matches.py
└── web/routes.py, templates/
scripts/setup_engines.py   # idempotent Maia-2 + Stockfish probe
scripts/smoke_3a.py        # H.6 two-user smoke test for Phase 3a
alembic/                   # Alembic migrations (Phase 3a+)
alembic.ini
Dockerfile, docker-compose.yml
tests/
```

## Upgrading from Phase 1

The Character schema gained three Elo columns. SQLAlchemy's `create_all()` doesn't alter existing tables, so:

```bash
rm metropolis_chess.db   # or delete the Docker volume `app-data`
```

Then restart. Presets reseed with the new values.

## Migrations (introduced in Phase 3a)

We use Alembic for schema migrations starting in 3a. Existing DBs should be stamped at the baseline and then upgraded; fresh DBs can either use the startup `create_all` path or `alembic upgrade head` from scratch.

**For a pre-3a local DB** (what most contributors have):
```bash
# From the repo root, with your .venv active
alembic stamp 0001_initial_baseline     # mark current state as the pre-3a baseline
alembic upgrade head                    # apply 0002 (Phase 3a: username + ownership + rating)
```
The 3a migration:
- Adds `players.username` (unique, NOT NULL); existing rows get `guest_<short_uuid>`.
- Adds `players.max_content_rating` (default `family`).
- Adds `characters.owner_id`, `visibility`, `content_rating`.
- Creates a `legacy_system` Player and assigns all pre-existing ownerless non-preset characters to them (so they stay accessible after ownership goes live).
- Applies preset ratings (Viktor + Kenji → mature; Margot + Archibald → family).

**Refreshing preset openings on an existing DB**: the seed logic skips already-seeded presets, so tweaks to a `PresetSpec.opening_preferences` do not flow to rows that exist. Use the helper script (safe to run any time; idempotent):

```bash
python -m scripts.refresh_preset_openings
```

It overwrites only `opening_preferences` on matching preset rows. Memories, Elo state, and voice are left alone.

**For a fresh DB**: `create_all` at app startup creates the full current schema. Stamp at head to record it:
```bash
alembic stamp head
```
Or, if you've deleted `metropolis_chess.db` and want Alembic as the sole creation path, run `alembic upgrade head` before starting the app — the 0001 baseline is a no-op, and 0002 is idempotent against already-materialized columns.

## Deploying to Hugging Face Spaces

This repo ships as a Docker SDK Space. Create a Space at `huggingface.co/new-space`, choose **Docker** as the SDK, then add this repo as the Space's git remote:

```bash
# Replace <username> and <space-name> with your HF username and chosen Space name.
git remote add hf https://huggingface.co/spaces/<username>/<space-name>
git push hf main
```

The first push triggers a build (10–20 min — Maia-2 + torch download). Watch progress in the Space's **Logs** tab.

### Space Secrets (set in Space Settings → Variables and Secrets)

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | **Yes** | Google Gemini API key for Kenji's Soul + memory generation. Without it the app runs but presets seed without memories and Kenji is silent. |
| `SESSION_SECRET` | Recommended | Arbitrary secret string for session cookie integrity. Defaults to empty (unsigned cookies — fine for a test deployment). |

### Variables with baked-in defaults (no action required)

| Variable | Default in image | Notes |
|---|---|---|
| `DATABASE_URL` | `sqlite:////data/metropolis_chess.db` | SQLite on HF persistent storage (`/data`). Survives Space restarts. |
| `REDIS_URL` | *(empty)* | No Redis on HF Spaces. The app falls back to in-process mood/cache state — fine for single-worker. |
| `STOCKFISH_PATH` | `/usr/games/stockfish` | Installed via `apt` in the image. |
| `MAIA2_CACHE_DIR` | `/app/maia2_models` | Weights pre-downloaded into the image at build time. |
| `LOG_DIR` | `/app/logs` | Logs also stream to stdout, which HF captures in the Logs tab. |

### WebSocket note

HF Spaces Docker supports WebSockets. The Socket.IO transport layer negotiates WebSocket by default and falls back to long-polling automatically — both paths work, WebSocket is faster. If you see `transport=polling` in the browser DevTools network tab, long-polling is active; gameplay still works but chat responsiveness is reduced.

## License

TBD.
