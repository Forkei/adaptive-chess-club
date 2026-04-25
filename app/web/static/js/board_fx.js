/* Phase 4.4c — board animation helpers.
 *
 * Uses Anime.js (loaded via CDN in play.html / pvp.html). Exposes
 * `window.MpBoardFx` with three primitives:
 *
 *   MpBoardFx.flashLastMove(uci)   — brass glow on source + dest squares
 *   MpBoardFx.pulseCheck(king_sq)  — red pulse on the checked king
 *   MpBoardFx.shakeBoard()         — board shakes on checkmate / flag
 *   MpBoardFx.slideBubble(el)      — slide-in the most recent chat bubble
 *
 * Squares are addressed by chessboard.js's `[data-square="e4"]` attribute.
 * The helpers tolerate Anime.js being absent (no-op).
 */
(function () {
  "use strict";
  const anime = window.anime;

  function sqEl(square) {
    return document.querySelector(`#board [data-square="${square}"]`);
  }

  function flashLastMove(uci) {
    if (!uci || uci.length < 4) return;
    const src = sqEl(uci.slice(0, 2));
    const dst = sqEl(uci.slice(2, 4));
    const applyGlow = (el, isDst) => {
      if (!el) return;
      // Overlay a highlight layer so we don't fight chessboard.js's own bg.
      el.style.position = 'relative';
      const layer = document.createElement('div');
      layer.className = 'mp-move-glow';
      layer.style.cssText = `
        position:absolute; inset:0;
        background: ${isDst ? 'rgba(201,166,107,0.55)' : 'rgba(201,166,107,0.25)'};
        pointer-events:none; z-index:1;
      `;
      el.appendChild(layer);
      if (anime) {
        anime({
          targets: layer,
          opacity: [1, 0],
          duration: isDst ? 1400 : 1000,
          easing: 'easeOutQuad',
          complete: () => layer.remove(),
        });
      } else {
        setTimeout(() => layer.remove(), 1200);
      }
    };
    applyGlow(src, false);
    applyGlow(dst, true);
  }

  function pulseCheck(kingSquare) {
    const el = sqEl(kingSquare);
    if (!el) return;
    el.style.position = 'relative';
    const layer = document.createElement('div');
    layer.style.cssText = `
      position:absolute; inset:0;
      background: rgba(181, 58, 54, 0.55);
      pointer-events:none; z-index:1;
      box-shadow: inset 0 0 24px rgba(231,158,155,0.9);
    `;
    el.appendChild(layer);
    if (anime) {
      anime({
        targets: layer,
        opacity: [
          { value: 0.9, duration: 180 },
          { value: 0,   duration: 320 },
          { value: 0.8, duration: 180 },
          { value: 0,   duration: 420 },
        ],
        easing: 'easeInOutQuad',
        complete: () => layer.remove(),
      });
    } else {
      setTimeout(() => layer.remove(), 1200);
    }
  }

  function shakeBoard() {
    const frame = document.querySelector('#board, .mp-board-frame');
    if (!frame) return;
    if (!anime) return;
    anime({
      targets: frame,
      translateX: [
        { value: -8, duration: 60 },
        { value:  8, duration: 60 },
        { value: -6, duration: 70 },
        { value:  5, duration: 80 },
        { value:  0, duration: 100 },
      ],
      easing: 'easeInOutSine',
    });
  }

  function slideBubble(el) {
    if (!el || !anime) return;
    const fromLeft = el.classList.contains('chat-bubble-agent');
    anime.set(el, { opacity: 0, translateX: fromLeft ? -20 : 20 });
    anime({
      targets: el,
      opacity: [0, 1],
      translateX: [fromLeft ? -20 : 20, 0],
      duration: 320,
      easing: 'easeOutCubic',
    });
  }

  window.MpBoardFx = Object.freeze({
    flashLastMove, pulseCheck, shakeBoard, slideBubble,
  });
})();
