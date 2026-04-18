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
| **2b**  | Subconscious (sqlite-vec retrieval), Soul (chat responses with tool-use), OpponentProfile, post-match processor | ⏳ next |
| **3**   | Real-time chat (Socket.IO), mood evolution from chat, disconnect handling, scheduler | ⏳ future |

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

### What's deliberately NOT in 2a

- No Subconscious, Soul, or post-match processor
- No chat field on moves, no surfaced memories in responses
- No `OpponentProfile` aggregation logic
- No Socket.IO / real-time streaming
- No scheduler / idle detection

## Setup

**Supported dev targets:** macOS, Linux, Windows via WSL2. The Dockerfile is the contract — that's where Stockfish, Maia-2, and all engine deps are known to work. Windows-native is not supported.

### Docker (recommended)

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
Dockerfile, docker-compose.yml
tests/
```

## Upgrading from Phase 1

The Character schema gained three Elo columns. SQLAlchemy's `create_all()` doesn't alter existing tables, so:

```bash
rm metropolis_chess.db   # or delete the Docker volume `app-data`
```

Then restart. Presets reseed with the new values.

## License

TBD.
