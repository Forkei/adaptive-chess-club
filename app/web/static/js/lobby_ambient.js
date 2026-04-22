/* Phase 4.4d — lobby ambient bed.
 *
 * "Slightly busy chess club": a real café-chatter loop as a foundation,
 * plus sparse random triggers of piece clicks and cup clinks so the
 * room feels alive. All layered through MpSfx / MpAmbient gain so the
 * single nav-bar toggle silences everything together.
 *
 * Design choices:
 * - Single foundation loop. Looping a single café recording sounds less
 *   unnatural than stitching clips together if it's long enough (~1m+).
 * - Gentle duck when the board page is reached (via mp-audio-duck).
 * - Accent sounds at low gain, randomly panned so they feel distant.
 */
(function () {
  "use strict";

  // Ambient audio disabled for the demo.
  return;

  // Only run on the lobby page.
  if (!document.body || !document.body.dataset.mpLobbyId) return;

  const MUTED_KEY = "mp_ambient_muted";
  const VOL_KEY = "mp_ambient_volume";
  const BED_PATH = "/static/audio/ambient/coffee_shop.ogg";

  function savedMuted() {
    const v = localStorage.getItem(MUTED_KEY);
    return v === null ? true : v === "true";
  }
  function savedVolume() {
    const v = parseFloat(localStorage.getItem(VOL_KEY));
    return Number.isFinite(v) ? v : 0.42;
  }

  const bed = document.createElement("audio");
  bed.id = "mp-lobby-bed";
  bed.src = BED_PATH;
  bed.loop = true;
  bed.preload = "auto";
  bed.volume = savedMuted() ? 0 : savedVolume();
  document.body.appendChild(bed);

  function attemptPlay() {
    bed.play().catch(() => { /* autoplay blocked — retry on first gesture */ });
  }
  function setBedMuted(muted) {
    const target = muted ? 0 : savedVolume();
    const start = bed.volume;
    const t0 = performance.now();
    const dur = 500;
    function step(t) {
      const k = Math.min(1, (t - t0) / dur);
      bed.volume = start + (target - start) * k;
      if (k < 1) requestAnimationFrame(step);
      else if (!muted) attemptPlay();
    }
    requestAnimationFrame(step);
  }

  // --- Accent layer: sparse random piece-clicks and clinks --------------
  // Fires every 3-9 seconds when active. Each trigger picks one SFX and
  // plays it at low volume. Gives the illusion of other games nearby.
  const ACCENTS = [
    { name: "piece_move",    volume: 0.14, rate: 0.95, rateJitter: 0.1 },
    { name: "piece_move",    volume: 0.11, rate: 1.05, rateJitter: 0.1 },
    { name: "piece_capture", volume: 0.10, rate: 0.95, rateJitter: 0.08 },
    { name: "chime",         volume: 0.06, rate: 1.0, rateJitter: 0.05 }, // cup clink
    { name: "page_flip",     volume: 0.10, rate: 1.0, rateJitter: 0.05 }, // notation being written
    { name: "creak",         volume: 0.09, rate: 0.9, rateJitter: 0.05 }, // chair
  ];
  let accentTimer = null;
  function scheduleAccent() {
    const delay = 3000 + Math.random() * 6000;
    accentTimer = setTimeout(() => {
      if (!savedMuted() && window.MpSfx) {
        const a = ACCENTS[Math.floor(Math.random() * ACCENTS.length)];
        const rate = a.rate + (Math.random() * 2 - 1) * a.rateJitter;
        window.MpSfx.play(a.name, { volume: a.volume, rate });
      }
      scheduleAccent();
    }, delay);
  }
  function startAccents() {
    if (!accentTimer) scheduleAccent();
  }
  function stopAccents() {
    if (accentTimer) { clearTimeout(accentTimer); accentTimer = null; }
  }

  // --- Wire to the global mute toggle -----------------------------------
  // ambient.js / synth_ambient.js writes mp_ambient_muted on click; we
  // poll at first click since we're not sharing a gain graph.
  document.addEventListener("click", (ev) => {
    const t = ev.target.closest && ev.target.closest("#mp-ambient-toggle");
    if (!t) return;
    setTimeout(() => {
      const muted = savedMuted();
      setBedMuted(muted);
      if (muted) stopAccents(); else startAccents();
    }, 0);
  }, true);

  // First user gesture: this is the only moment where browsers allow us
  // to start audio playback. Fix 4 (demo-rescue): if the user hasn't
  // explicitly chosen to mute (no localStorage entry), implicitly unmute
  // on first gesture so the lobby actually has its café chatter. A
  // deliberate click on the ♪ toggle still wins — we only flip when
  // there's no prior user preference on record.
  function onFirstGesture() {
    const explicit = localStorage.getItem(MUTED_KEY);
    if (explicit === null) {
      localStorage.setItem(MUTED_KEY, "false");
      bed.volume = savedVolume();
      const toggle = document.getElementById("mp-ambient-toggle");
      if (toggle) {
        toggle.dataset.muted = "false";
        toggle.setAttribute("aria-pressed", "true");
        toggle.innerHTML = '<span aria-hidden="true">♪</span>';
      }
    }
    if (!savedMuted()) { attemptPlay(); startAccents(); }
    window.removeEventListener("pointerdown", onFirstGesture, true);
    window.removeEventListener("keydown", onFirstGesture, true);
  }
  window.addEventListener("pointerdown", onFirstGesture, true);
  window.addEventListener("keydown", onFirstGesture, true);

  // Duck/unduck (e.g. if we show a video in-lobby later).
  document.addEventListener("mp-audio-duck", () => {
    if (savedMuted()) return;
    bed.volume = Math.max(0.06, savedVolume() * 0.35);
  });
  document.addEventListener("mp-audio-unduck", () => {
    if (savedMuted()) return;
    bed.volume = savedVolume();
  });

  // If the user is already unmuted from another page, bed starts as soon
  // as allowed. Initial attempt is likely to fail (autoplay) — accents
  // still stay idle until the gesture handler fires.
  if (!savedMuted()) attemptPlay();
})();
