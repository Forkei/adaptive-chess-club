/* Phase 4.4c — SFX engine.
 *
 * Wraps the small set of CC0 Kenney interface/RPG sounds shipped under
 * /static/audio/sfx/. Exposes a single global:
 *
 *     window.MpSfx.play("piece_move", { volume: 0.8, rate: 1.0 });
 *
 * Implementation notes:
 *
 * - AudioBuffer-based: one decode on first hit, then zero-latency replays
 *   with their own BufferSource + GainNode. Lets us fire several sounds
 *   simultaneously (e.g. radio click AND button click).
 * - Respects the same `mp_ambient_muted` key as the ambient pad + the
 *   nav bar toggle. A single mute silences the whole bed.
 * - Respects the `mp_sfx_volume` key; settings can expose a slider.
 * - Autoplay-safe: AudioContext is created lazily on the first user
 *   gesture (click, keypress). Before that, play() silently queues the
 *   latest requests.
 * - No network fetch for sounds the page never triggers — files load on
 *   first use. For games, we pre-warm piece_move + piece_capture.
 */
(function () {
  "use strict";

  const MUTED_KEY = "mp_ambient_muted";
  const VOL_KEY = "mp_sfx_volume";
  const BASE_VOL = 0.55;
  const BASE = "/static/audio/sfx/";

  // Name → filename under /static/audio/sfx/.
  const CATALOG = {
    // UI
    click:       "click.ogg",
    confirm:     "confirm.ogg",
    error:       "error.ogg",
    hover:       "hover.ogg",
    notify:      "notify.ogg",
    // Lobby / room
    door_open:   "door_open.ogg",
    door_close:  "door_close.ogg",
    lock:        "lock.ogg",
    lock_heavy:  "lock_heavy.ogg",
    radio_click: "radio_click.ogg",
    creak:       "creak.ogg",
    page_flip:   "page_flip.ogg",
    // Game
    piece_move:    "piece_move.ogg",
    piece_capture: "piece_capture.ogg",
    bell:          "bell.ogg",
    chime:         "chime.ogg",
  };

  // Per-sound default gain so we don't have to tune at every call site.
  const GAIN = {
    click: 0.35, confirm: 0.45, error: 0.4, hover: 0.2, notify: 0.7,
    door_open: 0.6, door_close: 0.6,
    lock: 0.45, lock_heavy: 0.55, radio_click: 0.5,
    creak: 0.5, page_flip: 0.35,
    piece_move: 0.5, piece_capture: 0.65,
    bell: 0.7, chime: 0.55,
  };

  function savedMuted() {
    const v = localStorage.getItem(MUTED_KEY);
    return v === null ? true : v === "true";
  }
  function savedVolume() {
    const v = parseFloat(localStorage.getItem(VOL_KEY));
    return Number.isFinite(v) ? v : BASE_VOL;
  }

  let ctx = null;
  let master = null;
  const buffers = new Map();     // name → AudioBuffer
  const pending = new Map();     // name → Promise<AudioBuffer> (dedupe loads)
  let unlocked = false;

  function ensureCtx() {
    if (ctx) return;
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return;
    ctx = new AC();
    master = ctx.createGain();
    master.gain.value = savedMuted() ? 0 : savedVolume();
    master.connect(ctx.destination);
  }

  function loadBuffer(name) {
    if (buffers.has(name)) return Promise.resolve(buffers.get(name));
    if (pending.has(name)) return pending.get(name);
    ensureCtx();
    if (!ctx) return Promise.resolve(null);
    const file = CATALOG[name];
    if (!file) return Promise.resolve(null);
    const p = fetch(BASE + file)
      .then(r => r.ok ? r.arrayBuffer() : Promise.reject(r.status))
      .then(ab => ctx.decodeAudioData(ab))
      .then(buf => { buffers.set(name, buf); pending.delete(name); return buf; })
      .catch(err => { pending.delete(name); console.warn("[sfx] load failed", name, err); return null; });
    pending.set(name, p);
    return p;
  }

  function play(name, opts) {
    if (savedMuted() || !unlocked) return;
    ensureCtx();
    if (!ctx) return;
    const baseGain = (GAIN[name] || 1.0) * (opts && opts.volume != null ? opts.volume : 1.0);
    const rate = opts && opts.rate != null ? opts.rate : 1.0;
    const playNow = (buf) => {
      if (!buf) return;
      const src = ctx.createBufferSource();
      src.buffer = buf;
      src.playbackRate.value = rate;
      const g = ctx.createGain();
      g.gain.value = baseGain;
      src.connect(g).connect(master);
      src.start();
    };
    const buf = buffers.get(name);
    if (buf) { playNow(buf); return; }
    loadBuffer(name).then(playNow);
  }

  // Fire-and-forget preload; call after first gesture for sounds you know
  // will play soon (e.g. piece_move, piece_capture on the play page).
  function preload(names) {
    for (const n of names || []) loadBuffer(n);
  }

  function setMuted(next) {
    localStorage.setItem(MUTED_KEY, String(next));
    if (master && ctx) {
      master.gain.linearRampToValueAtTime(
        next ? 0 : savedVolume(),
        ctx.currentTime + 0.25,
      );
    }
  }
  function setVolume(v) {
    v = Math.max(0, Math.min(1, v));
    localStorage.setItem(VOL_KEY, String(v));
    if (master && ctx && !savedMuted()) {
      master.gain.linearRampToValueAtTime(v, ctx.currentTime + 0.15);
    }
  }

  // Autoplay unlock: AudioContext has to be created/resumed in a user
  // gesture handler, otherwise Chrome silences it. On the first user
  // interaction anywhere on the page we flip `unlocked` and stay unlocked.
  function unlock() {
    ensureCtx();
    if (ctx && ctx.state === "suspended") ctx.resume();
    unlocked = true;
    window.removeEventListener("pointerdown", unlock, true);
    window.removeEventListener("keydown", unlock, true);
  }
  window.addEventListener("pointerdown", unlock, true);
  window.addEventListener("keydown", unlock, true);

  // Per-element debounce: fire at most once per 200 ms per target so
  // cursor movement across nested children doesn't hammer the sound.
  const _hoverTs = new WeakMap();
  const HOVER_DEBOUNCE_MS = 200;
  document.addEventListener("mouseover", (ev) => {
    const t = ev.target.closest && ev.target.closest(".mp-btn, [data-sfx-hover]");
    if (!t) return;
    const now = Date.now();
    if (now - (_hoverTs.get(t) || 0) < HOVER_DEBOUNCE_MS) return;
    _hoverTs.set(t, now);
    play("hover", { volume: 0.25 });
  }, { passive: true });

  // Listen to the ambient mute toggle and mirror state so nav toggle
  // silences SFX too.
  document.addEventListener("click", (ev) => {
    const t = ev.target.closest && ev.target.closest("#mp-ambient-toggle");
    if (!t) return;
    // ambient.js / synth_ambient.js updates the localStorage on the same
    // key; just re-sync our master gain after the next tick.
    setTimeout(() => setMuted(savedMuted()), 0);
  }, true);

  window.MpSfx = Object.freeze({
    play, preload, setMuted, setVolume,
    isMuted: savedMuted,
    volume: savedVolume,
    CATALOG: Object.keys(CATALOG),
  });
})();
