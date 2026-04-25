/* Phase 4.1c — emotion video player.
 *
 * Contract: the play page renders
 *   <div id="mp-emotion-stage" data-clips='{"neutral": "/static/...neutral.mp4", ...}'>
 *     <video id="mp-emotion-a" muted playsinline preload="auto" loop></video>
 *     <video id="mp-emotion-b" muted playsinline preload="auto" loop></video>
 *   </div>
 *
 * When code elsewhere calls `MpEmotion.set('pleased', 0.7)`, we look up
 * the corresponding clip URL. If it's missing, we fall back to the
 * `neutral` clip. If that's missing too, the whole stage stays hidden
 * and the text-only indicator takes over (handled by inline code on the
 * play page).
 *
 * Two <video> tags are kept alternating so we can crossfade between
 * clips with non-seamless start frames (user's Viktor clips have
 * mismatched first frames by design).
 */
(function () {
  const stage = document.getElementById("mp-emotion-stage");
  if (!stage) return;

  const a = document.getElementById("mp-emotion-a");
  const b = document.getElementById("mp-emotion-b");
  if (!a || !b) return;

  let clips = {};
  try {
    clips = JSON.parse(stage.dataset.clips || "{}");
  } catch (_) {
    clips = {};
  }

  let currentEl = a;
  let currentEmotion = null;

  function resolveUrl(emotion) {
    if (clips[emotion]) return clips[emotion];
    if (clips.neutral) return clips.neutral;
    return null;
  }

  // Preload check: briefly try to load and see if we get metadata. If the
  // file 404s we hide the stage permanently so the text indicator takes over.
  function probeNeutral() {
    const url = clips.neutral;
    if (!url) {
      stage.style.display = "none";
      return;
    }
    const probe = new Audio();  // any element supporting HEAD via src works
    const probeVideo = document.createElement("video");
    probeVideo.preload = "metadata";
    probeVideo.muted = true;
    let settled = false;
    const onErr = () => {
      if (settled) return;
      settled = true;
      stage.style.display = "none";
    };
    const onOk = () => {
      if (settled) return;
      settled = true;
      setEmotion("neutral", 1.0);
    };
    probeVideo.addEventListener("loadedmetadata", onOk, { once: true });
    probeVideo.addEventListener("error", onErr, { once: true });
    probeVideo.src = url;
  }

  // Monotonic version counter. probeNeutral and loadHistory often call
  // setEmotion within the same tick; without this, two play() handlers
  // race on the same <video> element, both fire on canplaythrough, and
  // the second pauses the now-visible element on a dark frame.
  let setVersion = 0;

  function setEmotion(emotion, intensity) {
    if (emotion === currentEmotion) return;
    const url = resolveUrl(emotion);
    if (!url) return;

    const myVersion = ++setVersion;
    const nextEl = currentEl === a ? b : a;
    nextEl.src = url;
    nextEl.style.opacity = "0";
    nextEl.style.transition = "opacity 420ms ease";
    nextEl.load();

    const play = () => {
      // Stale: a newer setEmotion has superseded us. Don't touch opacity
      // or pause anything — let the latest call drive the swap.
      if (myVersion !== setVersion) return;
      nextEl.play().catch(() => {});
      // Swap visible element.
      requestAnimationFrame(() => {
        if (myVersion !== setVersion) return;
        nextEl.style.opacity = "1";
        currentEl.style.opacity = "0";
      });
      setTimeout(() => {
        if (myVersion !== setVersion) return;
        currentEl.pause();
        currentEl = nextEl;
        currentEmotion = emotion;
      }, 480);
    };
    nextEl.addEventListener("canplaythrough", play, { once: true });
    // Fallback: if canplaythrough never fires, show after a timeout.
    setTimeout(() => {
      if (myVersion !== setVersion) return;
      if (nextEl.readyState >= 2 && currentEl !== nextEl) play();
    }, 800);
  }

  // Style bootstrap: common to both video tags.
  [a, b].forEach((v) => {
    v.muted = true;
    v.playsInline = true;
    v.loop = true;
    v.style.position = "absolute";
    v.style.inset = "0";
    v.style.width = "100%";
    v.style.height = "100%";
    v.style.objectFit = "cover";
    // Fix 2: if the template already rendered a <video src="..."> inline
    // (so the stage isn't blank on first paint before MpEmotion boots),
    // keep that element visible and mark it as the current element so the
    // probe's crossfade picks the empty partner, not the already-playing one.
    if (v.getAttribute("src")) {
      v.style.opacity = "1";
      currentEl = v;
      currentEmotion = "neutral";
      v.play().catch(() => {});
    } else {
      v.style.opacity = "0";
    }
    v.style.transition = "opacity 420ms ease";
  });
  stage.style.position = stage.style.position || "relative";

  window.MpEmotion = Object.freeze({
    set: setEmotion,
    probe: probeNeutral,
  });

  // The play page (Fix 2) already rendered video `a` inline with a real src,
  // so the stage is never blank on first paint. Running probeNeutral() in
  // that case is a no-op on success (early-return: currentEmotion already
  // "neutral") but a stage-killer on any error — the probe's onErr path
  // calls stage.style.display = "none", wiping an already-working video.
  // Instead, attach an error listener directly to the inline element so
  // failures degrade cleanly.  On the detail page (no inline src) the probe
  // still runs as designed to load the first clip.
  if (currentEl.getAttribute("src")) {
    currentEl.addEventListener("error", () => {
      stage.style.display = "none";
    }, { once: true });
  } else {
    probeNeutral();
  }
})();
