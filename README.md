# Metropolis Chess Club

A chess-playing AI character platform. Users create or pick an AI opponent — each with its own backstory, personality sliders, opening preferences, and lived-in memories — then play against it with real-time chat.

The runtime has three layers (Phase 2+):

- **Soul** — an LLM that handles chat, personality, and intent
- **Director** — deterministic code that translates the Soul's mood/intent into engine config
- **Body** — a chess engine (Maia-2 primary, Stockfish for high-skill "beast mode")

A **Subconscious** agent retrieves relevant memories every turn and feeds them to the Soul.

## Phase 1 status (this repo)

Phase 1 builds the **character foundation** only — no engine, no gameplay, no real-time chat yet.

### What's implemented

- FastAPI app with SQLAlchemy 2.0 (SQLite dev) and Jinja2 templates
- `Character` schema with sliders (aggression, risk_tolerance, patience, trash_talk), ELO + adaptive flag, opening preferences (curated ECO list with group tags), voice/tone, quirks
- `Memory` schema with scope/type enums, emotional valence, triggers, relevance tags, surface tracking
- `style_to_prompt_fragments` — deterministic slider → prompt snippet mapping (used by Soul in Phase 2)
- Curated ~40-opening list with group tags (`king_pawn_open`, `queen_pawn_closed`, `flank`, `gambit`, `indian`, `hypermodern`, `unorthodox`) for future Director logic
- Single `app/llm/client.py` wrapper around `google-genai` with Pydantic-schema structured output, retries, and JSONL call logging
- Backstory → ~40 first-person memories generator (async via `BackgroundTasks`)
- Character state machine: `generating_memories` → `ready` / `generation_failed` (with error message + started-at timestamp)
- 4 preset characters (Viktor Volkov, Margot Lindqvist, Kenji Sato, Archibald Finch), idempotently seeded on startup
- REST API: list / detail / create / delete / browse memories
- Plain HTML UI with Tailwind CDN: character grid, create form, detail page with auto-refresh while generating
- Tests: style helper, memory schema validation, memory generator (mocked), preset seeding idempotency, opt-in live LLM test

### What's NOT implemented yet (Phase 2+)

- Maia-2 / Stockfish integration
- The Soul / Director / Subconscious runtime
- Socket.IO / real-time chat
- Matches, move validation, gameplay
- Mood state, disconnect handling, post-game processing
- Redis / multi-process deployment

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |    macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env    # then edit GEMINI_API_KEY

uvicorn app.main:app --reload
```

Open http://localhost:8000 — you should see 4 preset characters (they'll appear as `generating_memories` for ~30–60s on first startup, then switch to `ready`).

## Running tests

```bash
pytest                                   # mocks only; fast
RUN_LIVE_LLM_TESTS=1 pytest -m live      # opt-in, needs GEMINI_API_KEY
```

## Layout

```
app/
├── main.py               # FastAPI entry, startup seeding
├── config.py             # pydantic-settings
├── db.py                 # SQLAlchemy 2.0 engine + session
├── logging_config.py
├── llm/
│   ├── client.py         # google-genai wrapper
│   └── call_log.py       # JSONL call log
├── models/               # SQLAlchemy ORM (Character, Memory)
├── schemas/              # Pydantic v2 (create/read/summary)
├── characters/
│   ├── style.py          # style_to_prompt_fragments
│   ├── openings.py       # curated opening list + groups
│   ├── memory_generator.py
│   ├── presets.py
│   └── seed.py
├── memory/crud.py
├── api/characters.py
└── web/routes.py, templates/
tests/
```

## License

TBD.
