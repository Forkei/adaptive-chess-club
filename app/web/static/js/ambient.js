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
 */
(function () {
  const MUTED_KEY = "mp_ambient_muted";
  const VOL_KEY = "mp_ambient_volume";
  const BASE_VOL = 0.45;

  const audio = document.getElementById("mp-ambient");
  const toggle = document.getElementById("mp-ambient-toggle");
  if (!audio) return;

  function savedMuted() {
    // Default to muted (respect autoplay policy) unless user explicitly unmuted.
    const v = localStorage.getItem(MUTED_KEY);
    return v === null ? true : v === "true";
  }
  function savedVolume() {
    const v = parseFloat(localStorage.getItem(VOL_KEY));
    return Number.isFinite(v) ? v : BASE_VOL;
  }

  let muted = savedMuted();
  audio.volume = savedVolume();
  audio.muted = muted;

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
