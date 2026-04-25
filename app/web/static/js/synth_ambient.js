/* Phase 4.4b — synthesised ambient audio.
 *
 * The character room assets never shipped. Rather than waiting for them,
 * we generate a soft, slowly-evolving drone pad in the browser via
 * WebAudio. Each room's pad is keyed on `body[data-theme]` so Viktor's
 * room feels minor/bronze-deep, Margot's airy/major, etc. Zero file
 * assets required.
 *
 * Contract: mirrors `ambient.js` — respects the same `#mp-ambient-toggle`
 * button if present, persists mute state to `mp_ambient_muted` in
 * localStorage. Autoplay-safe: starts silent, only opens the gain on
 * user gesture.
 */
(function () {
  "use strict";
  const MUTED_KEY = "mp_ambient_muted";
  const VOL_KEY = "mp_ambient_volume";
  const BASE_VOL = 0.28;

  // If a real <audio> element is present, let ambient.js drive it and
  // bail out. The two scripts are mutually exclusive.
  if (document.getElementById("mp-ambient")) return;
  // Lobby uses its own layered ambient bed (lobby_ambient.js). Don't
  // double up with a synth pad underneath.
  if (document.body && document.body.dataset && document.body.dataset.mpLobbyId) return;

  const toggle = document.getElementById("mp-ambient-toggle");
  const theme = (document.body && document.body.dataset && document.body.dataset.theme) || "default";

  // --- Per-room chord voicings (midi note numbers). Slow root + gently
  //     detuned triads, plus a high shimmer partial. All quiet enough to
  //     sit under voice + UI without competing.
  const VOICINGS = {
    // Viktor: minor, cold, wide-spaced.
    viktor_petrov: { root: 45, intervals: [0, 7, 10, 15, 22], detune: 6, filter: 520, wobble: 0.13 },
    // Margot: airy major seventh, bright shimmer.
    margot_lindqvist: { root: 52, intervals: [0, 4, 7, 11, 19], detune: 4, filter: 900, wobble: 0.08 },
    // Kenji: perfect fifths, modal, steady.
    kenji_sato: { root: 48, intervals: [0, 7, 12, 19], detune: 3, filter: 700, wobble: 0.05 },
    // Archibald: warm minor 9th, wood + leather.
    archibald_finch: { root: 43, intervals: [0, 3, 7, 10, 14], detune: 5, filter: 600, wobble: 0.09 },
    default: { root: 48, intervals: [0, 7, 12, 16], detune: 4, filter: 700, wobble: 0.07 },
  };

  const spec = VOICINGS[theme] || VOICINGS.default;

  function savedMuted() {
    const v = localStorage.getItem(MUTED_KEY);
    return v === null ? true : v === "true";
  }
  function savedVolume() {
    const v = parseFloat(localStorage.getItem(VOL_KEY));
    return Number.isFinite(v) ? v : BASE_VOL;
  }
  function mtof(m) { return 440 * Math.pow(2, (m - 69) / 12); }

  let muted = savedMuted();
  let ctx = null;
  let master = null;
  let started = false;

  function renderToggle() {
    if (!toggle) return;
    toggle.dataset.muted = String(muted);
    toggle.setAttribute("aria-pressed", String(!muted));
    toggle.title = muted ? "Unmute room audio" : "Mute room audio";
    toggle.innerHTML = muted
      ? '<span aria-hidden="true">🔇</span>'
      : '<span aria-hidden="true">🔊</span>';
  }

  function ensureStarted() {
    if (started) return;
    try {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) return;
      ctx = new AC();
      master = ctx.createGain();
      master.gain.value = muted ? 0 : savedVolume();
      // Very gentle global lowpass so the whole bed feels distant.
      const shelf = ctx.createBiquadFilter();
      shelf.type = "lowpass";
      shelf.frequency.value = spec.filter * 2;
      shelf.Q.value = 0.7;
      master.connect(shelf).connect(ctx.destination);

      // One oscillator per interval, slightly detuned + panned for width.
      for (let i = 0; i < spec.intervals.length; i++) {
        const interval = spec.intervals[i];
        const freq = mtof(spec.root + interval);
        // Two detuned sawtooths + a sine for body = simple "super saw" pad.
        const voiceGain = ctx.createGain();
        voiceGain.gain.value = 0;
        const pan = ctx.createStereoPanner
          ? ctx.createStereoPanner()
          : null;
        if (pan) {
          pan.pan.value = (i - (spec.intervals.length - 1) / 2) * 0.5;
          voiceGain.connect(pan).connect(master);
        } else {
          voiceGain.connect(master);
        }

        const saw1 = ctx.createOscillator();
        saw1.type = "sawtooth";
        saw1.frequency.value = freq;
        saw1.detune.value = -spec.detune;
        const saw2 = ctx.createOscillator();
        saw2.type = "sawtooth";
        saw2.frequency.value = freq;
        saw2.detune.value = spec.detune;
        const sine = ctx.createOscillator();
        sine.type = "sine";
        sine.frequency.value = freq / 2;

        const filter = ctx.createBiquadFilter();
        filter.type = "lowpass";
        filter.frequency.value = spec.filter;
        filter.Q.value = 1.4;

        saw1.connect(filter);
        saw2.connect(filter);
        sine.connect(filter);
        filter.connect(voiceGain);

        // Slow LFO wobble on the voice gain to keep the pad breathing.
        const lfo = ctx.createOscillator();
        lfo.type = "sine";
        lfo.frequency.value = 0.05 + i * 0.017;
        const lfoGain = ctx.createGain();
        lfoGain.gain.value = spec.wobble * 0.09;
        lfo.connect(lfoGain).connect(voiceGain.gain);

        // Fade voice in over 4s.
        const t0 = ctx.currentTime;
        voiceGain.gain.setValueAtTime(0, t0);
        voiceGain.gain.linearRampToValueAtTime(0.09, t0 + 4);

        saw1.start(); saw2.start(); sine.start(); lfo.start();
      }
      started = true;
    } catch (_) {
      started = false;
    }
  }

  function setMuted(next) {
    muted = next;
    localStorage.setItem(MUTED_KEY, String(muted));
    if (!muted) {
      ensureStarted();
      if (ctx && ctx.state === "suspended") ctx.resume();
      if (master) master.gain.linearRampToValueAtTime(savedVolume(), ctx.currentTime + 0.4);
    } else if (master && ctx) {
      master.gain.linearRampToValueAtTime(0, ctx.currentTime + 0.4);
    }
    renderToggle();
  }

  renderToggle();
  if (toggle) toggle.addEventListener("click", () => setMuted(!muted));

  // On first user gesture, (re)start the context. Browsers require this.
  // Fix 4 (demo-rescue): if there's no stored mute preference, also flip
  // muted → false so the room actually has a pad. Explicit toggle clicks
  // still win because they set the localStorage entry.
  function onFirstGesture() {
    if (localStorage.getItem(MUTED_KEY) === null) setMuted(false);
    if (!muted) {
      ensureStarted();
      if (ctx && ctx.state === "suspended") ctx.resume();
    }
    window.removeEventListener("pointerdown", onFirstGesture, true);
    window.removeEventListener("keydown", onFirstGesture, true);
  }
  window.addEventListener("pointerdown", onFirstGesture, true);
  window.addEventListener("keydown", onFirstGesture, true);

  // Duck/unduck — same contract as ambient.js so play.html hooks work.
  document.addEventListener("mp-audio-duck", () => {
    if (!master || muted) return;
    master.gain.linearRampToValueAtTime(savedVolume() * 0.35, ctx.currentTime + 0.2);
  });
  document.addEventListener("mp-audio-unduck", () => {
    if (!master || muted) return;
    master.gain.linearRampToValueAtTime(savedVolume(), ctx.currentTime + 0.4);
  });

  window.MpAmbient = Object.freeze({
    mute: () => setMuted(true),
    unmute: () => setMuted(false),
    isMuted: () => muted,
  });
})();
