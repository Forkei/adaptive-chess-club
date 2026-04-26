/* Phase 4.1b — ambient-audio controller.
 *
 * Contract:
 *   A page opts in by rendering:
 *     <audio id="mp-ambient" src="/static/..." loop preload="auto"></audio>
 *     <button id="mp-ambient-toggle" data-muted="true">...</button>
 *
 * Browsers block autoplay of audio without a user gesture. We respect
 * that: start muted, wait for the user to click the toggle or any other
 * button, then unmute. Mute state is persisted to localStorage under
 * `mp_ambient_muted` so it survives navigation.
 *
 * Playback position is saved to sessionStorage on pagehide so the track
 * continues from where it left off when the user navigates to another
 * page that uses the same src. Different tracks always start fresh.
 */
(function () {
  const MUTED_KEY = "mp_ambient_muted";
  const VOL_KEY = "mp_ambient_volume";
  const POS_KEY = "mp_ambient_pos";  // { src, time }

  const audio = document.getElementById("mp-ambient");
  const toggle = document.getElementById("mp-ambient-toggle");
  if (!audio) return;

  // Per-page volume default: base.html sets data-ambient-volume when the
  // RoomTheme specifies one (e.g. menu page is quieter than game page).
  const _dataVol = parseFloat(audio.dataset.ambientVolume);
  const BASE_VOL = Number.isFinite(_dataVol) ? _dataVol : 0.45;

  function savedMuted() {
    // Default to muted (respect autoplay policy) unless user explicitly unmuted.
    const v = localStorage.getItem(MUTED_KEY);
    return v === null ? true : v === "true";
  }
  function savedVolume() {
    const v = parseFloat(localStorage.getItem(VOL_KEY));
    return Number.isFinite(v) ? v : BASE_VOL;
  }

  // Restore playback position if this page is loading the same track.
  function savedPos() {
    try {
      const raw = sessionStorage.getItem(POS_KEY);
      if (!raw) return null;
      const obj = JSON.parse(raw);
      // Normalise src for comparison: strip origin so relative/absolute URLs match.
      const norm = (u) => u.replace(/^https?:\/\/[^/]+/, "");
      if (norm(obj.src) === norm(audio.src)) return obj.time;
    } catch (_) {}
    return null;
  }

  // Save current position keyed to the track src before navigation.
  function savePos() {
    if (!audio.src || audio.duration === 0 || !isFinite(audio.currentTime)) return;
    try {
      sessionStorage.setItem(POS_KEY, JSON.stringify({ src: audio.src, time: audio.currentTime }));
    } catch (_) {}
  }
  window.addEventListener("pagehide", savePos);
  // visibilitychange covers tab-switch on mobile browsers that skip pagehide.
  document.addEventListener("visibilitychange", () => { if (document.visibilityState === "hidden") savePos(); });

  let muted = savedMuted();
  audio.volume = savedVolume();
  audio.muted = muted;

  // Seek to the saved position once the audio is ready to accept currentTime writes.
  const pos = savedPos();
  if (pos !== null && pos > 0) {
    audio.addEventListener("loadedmetadata", () => {
      if (pos < audio.duration) audio.currentTime = pos;
    }, { once: true });
    // If metadata is already loaded (from cache), set immediately.
    if (audio.readyState >= 1 && pos < audio.duration) audio.currentTime = pos;
  }

  function render() {
    if (!toggle) return;
    toggle.dataset.muted = String(muted);
    toggle.setAttribute("aria-pressed", String(!muted));
    toggle.title = muted ? "Unmute room audio" : "Mute room audio";
    toggle.innerHTML = muted
      ? '<span aria-hidden="true">🔇</span>'
      : '<span aria-hidden="true">🔊</span>';
  }

  function startPlaying() {
    audio.play().catch(() => {
      // Autoplay blocked. Will be retried on next user gesture.
    });
  }

  function setMuted(next) {
    muted = next;
    audio.muted = muted;
    localStorage.setItem(MUTED_KEY, String(muted));
    if (!muted) startPlaying();
    render();
  }

  render();
  startPlaying();

  if (toggle) {
    toggle.addEventListener("click", () => setMuted(!muted));
  }

  // One-shot: on first user gesture anywhere, (re)try starting playback so
  // browsers that blocked initial autoplay pick up once the user interacts.
  // Fix 4 (demo-rescue): if the user has no stored mute preference, also
  // implicitly unmute — otherwise the audio plays silently and looks
  // broken.
  //
  // Race-guard: if the first gesture IS a click on the mute toggle, skip
  // the auto-unmute and let the toggle's click handler win. Without this,
  // pointerdown fires onFirstGesture (setMuted(false)) and then the click
  // event fires the toggle handler (setMuted(true)) — net result: muted
  // and the button appears to do nothing.
  function onFirstGesture(ev) {
    const onToggle = ev && ev.target && typeof ev.target.closest === "function"
      && ev.target.closest("#mp-ambient-toggle");
    if (!onToggle && localStorage.getItem(MUTED_KEY) === null) {
      setMuted(false);  // auto-unmute; startPlaying() called inside
    } else if (!muted) {
      startPlaying();   // already unmuted from prior session — (re)start
    }
    window.removeEventListener("pointerdown", onFirstGesture, true);
    window.removeEventListener("keydown", onFirstGesture, true);
  }
  window.addEventListener("pointerdown", onFirstGesture, true);
  window.addEventListener("keydown", onFirstGesture, true);

  // Duck volume while the agent is thinking on the play page — keeps the
  // character's emotion clip / move audio from being drowned.
  document.addEventListener("mp-audio-duck", () => {
    if (!audio || audio.muted) return;
    audio.volume = Math.max(0.08, savedVolume() * 0.35);
  });
  document.addEventListener("mp-audio-unduck", () => {
    audio.volume = savedVolume();
  });

  // Expose minimal controls on window so inline scripts (e.g. play page
  // on game-over) can flip state without re-implementing storage keys.
  window.MpAmbient = Object.freeze({
    mute: () => setMuted(true),
    unmute: () => setMuted(false),
    isMuted: () => muted,
  });
})();
