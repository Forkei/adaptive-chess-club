# Phase 3b — manual browser smoke runbook

The automated Socket.IO tests (`tests/test_sockets_integration.py`) cover the server-side
behaviour with an in-process `socketio.AsyncClient`. This runbook is the complementary
browser check — it catches real-browser quirks (reconnect storms, tab backgrounding,
polling fallback) that the AsyncClient does not exercise.

## Prereqs

- Docker image built, or a local environment with `stockfish` + `maia2` installed.
- `GEMINI_API_KEY` set in `.env` (otherwise the agents go silent and `agent_chat` never fires).
- Server running on <http://localhost:8000>.
- DevTools open with the **Network** panel filtered to **WS** so you can watch the socket
  frames directly, and the **Console** open to see banner / error surfaces.

## M.6 — disconnect → reconnect within window resumes the match

1. Log in as any user; start a match from the character grid.
2. Once the board renders, play one move (e.g. `e2e4`). Wait for `agent_move` + `agent_chat`
   to come back — you should see the chat bubble + memory ribbon.
3. In DevTools **Application → Cookies**, confirm the `player_id` cookie exists.
4. **Close the tab.** (Not reload — a true tab close severs the WebSocket.)
5. Within ~2 minutes (plenty of margin under the 5-minute default cooldown), **re-open
   the match URL in a new tab** (`/matches/<id>`).
6. Verify: the page loads, the connection pill flips from `connecting…` → `live`, you see
   a brief `match_resumed` frame on the Network panel, and the disconnect overlay never
   appears.
7. Make another move — play continues normally.

**Pass criteria:** no data loss, no abandoned status, chat history and move list persisted.

## M.7 — disconnect → timeout → match abandoned

1. Temporarily set `MATCH_DISCONNECT_COOLDOWN_SECONDS=10` in `.env` and restart the
   server (so you don't wait 5 minutes).
2. Start a fresh match, play one move, get an `agent_move` response.
3. Close the tab. Do **not** reopen.
4. Wait at least 15 seconds. Then open `/matches/<id>` again.
5. Verify: the page loads with the **Finished / abandoned** status; the post-match
   panel appears and progresses through its steps; you get auto-redirected to
   `/matches/<id>/summary` within a few seconds.
6. Check `logs/` (or stdout) for `Disconnect cooldown started` followed by
   `disconnect_timeout` handling. Confirm no stack traces.

**Pass criteria:** match status is `abandoned`, post-match memories/narrative generated,
redirect works.

**Cleanup:** restore `MATCH_DISCONNECT_COOLDOWN_SECONDS` (delete the override or set back
to 300) before further testing.

## M.8 — chat-during-thinking flows into next Subconscious call

1. Start a match. Play `e2e4` (or any opening move).
2. The instant you see the **thinking…** banner (before `agent_move` returns), type a
   distinctive sentence in the chat box such as *"I am about to fianchetto my bishop and
   you won't see it coming"* and press Enter.
3. Verify: the bubble appears immediately with a faded ("in-transit") look, then
   de-fades on `player_chat_echoed`.
4. Wait for the character's response this turn — it doesn't have to reference your chat
   (the Soul might not speak at all).
5. Play another move. The **next** `agent_chat` response is the one that should reflect
   the mid-think chat (the Subconscious draining the pending buffer happens on the turn
   *after* the chat arrives).
6. Optional: tail the server logs for `pending_player_chat` debug lines to confirm the
   buffer was populated and drained.

**Pass criteria:** the chat bubble round-trips; the character's response within 2 turns
shows it received the message (via chat or behavioural cue — don't over-specify the LLM).

## General sanity / connection pill

- During any of the above, flip your Wi-Fi off briefly (≤15 s). The pill should flicker
  through `reconnecting…` and settle back on `live`. The disconnect overlay appears
  only after the WebSocket actually drops, and the server's cooldown doesn't start until
  then.
- Refreshing the page mid-game should also work — the new socket reconnects, the server
  cancels any armed cooldown task, and the client requests `match_state` via the
  connection handshake.

## If something is wrong

- **Overlay never hides after reconnect** — check that `match_resumed` fires in the
  Network panel. If not, the server didn't see the reconnect before the cooldown
  expired; match is probably already abandoned.
- **Post-match panel never advances** — check server logs for `post_match_status`
  emits. If the processor thread runs but emits never reach the socket, the
  `asyncio.run_coroutine_threadsafe` bridge in `app/sockets/bridge.py` may have
  captured `None` as the loop (app not fully started yet). A clean restart usually
  fixes this.
- **`agent_chat` never arrives** — the Soul returned `speak=None`. That's usually
  correct; most moves are silent. Watch `logs/llm_calls.jsonl` for a `soul:*` entry
  with `speak: null` to confirm.
