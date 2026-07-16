// Lightweight styled tooltips. Add data-tip="..." to any element (works for
// dynamically rendered content via event delegation). Tips show on mouse
// hover AND on keyboard focus, and their text is live: if the element's
// data-tip attribute changes while its tip is up (e.g. an updating readout),
// the tip re-reads and repositions.

let tipEl = null;
let showTimer = null;
let currentTarget = null;
let tipObserver = null; // watches the current target's data-tip for live edits

export function initTooltips() {
  tipEl = document.createElement('div');
  tipEl.id = 'tip';
  document.body.appendChild(tipEl);
  tipObserver = new MutationObserver(() => {
    if (currentTarget && tipEl.classList.contains('show')) show(currentTarget);
  });

  const tipTarget = n => n instanceof Element ? n.closest('[data-tip]') : null;
  const enter = t => {
    if (t === currentTarget) return;
    hide();
    if (!t) return;
    currentTarget = t;
    showTimer = setTimeout(() => show(t), 400);
  };
  const leave = (t, next) => {
    if (t && t === currentTarget && !(next instanceof Element && t.contains(next))) hide();
  };

  document.addEventListener('mouseover', e => enter(tipTarget(e.target)));
  document.addEventListener('mouseout', e => leave(tipTarget(e.target), e.relatedTarget));
  // keyboard: focused elements (buttons, tabindexed surfaces) get the same
  // tips — but only when the focus is keyboard-driven (:focus-visible
  // semantics). Clicks also move focus (buttons, and the tabindexed
  // containers via any non-focusable child), and without this gate the
  // focusin that follows every mousedown would resurrect the very tip the
  // mousedown just hid, parking it over whatever the user clicked.
  let keyboardFocus = false;
  document.addEventListener('keydown', () => { keyboardFocus = true; }, true);
  document.addEventListener('pointerdown', () => { keyboardFocus = false; }, true);
  document.addEventListener('focusin', e => { if (keyboardFocus) enter(tipTarget(e.target)); });
  document.addEventListener('focusout', e => leave(tipTarget(e.target), e.relatedTarget));
  document.addEventListener('mousedown', hide, true);
  window.addEventListener('scroll', hide, true);
}

function show(t) {
  if (!document.body.contains(t)) return hide();
  const text = t.dataset.tip;
  if (!text) return hide();
  tipEl.textContent = text;
  tipEl.classList.add('show');
  // live content: re-invoke show() when this element's data-tip changes
  // (re-observing the same node just replaces the options — no duplicates)
  tipObserver.observe(t, { attributes: true, attributeFilter: ['data-tip'] });
  const r = t.getBoundingClientRect();
  const tw = tipEl.offsetWidth;
  const th = tipEl.offsetHeight;
  const x = Math.min(Math.max(8, r.left + r.width / 2 - tw / 2), window.innerWidth - tw - 8);
  let y = r.bottom + 9;
  if (y + th > window.innerHeight - 8) y = r.top - th - 9;
  tipEl.style.left = `${x}px`;
  tipEl.style.top = `${y}px`;
}

function hide() {
  clearTimeout(showTimer);
  currentTarget = null;
  if (tipObserver) tipObserver.disconnect();
  if (tipEl) tipEl.classList.remove('show');
}
