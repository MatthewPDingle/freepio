// App bootstrap: tabs, setup form, solve dashboard, browser wiring.

import { api, toast } from './api.js';
import { RangeEditor } from './range_editor.js';
import { Browser, cardChip } from './browse.js';
import { RANKS, SUITS, SUIT_GLYPH, cardToString } from './cards.js';
import { initTooltips } from './tooltip.js';
import * as preflop from './preflop.js';

const $ = id => document.getElementById(id);
initTooltips();
$('status-pill').dataset.tip =
  'Solver state: idle (no tree) → ready (tree built) → running → done/stopped. The iteration counter ticks while solving.';

// ---------------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------------

const tabs = document.querySelectorAll('.tab');
function showTab(name) {
  tabs.forEach(t => t.classList.toggle('active', t.dataset.tab === name));
  document.querySelectorAll('.view').forEach(v =>
    v.classList.toggle('active', v.id === `view-${name}`));
  if (name === 'browse' && state.solved) browser.refresh();
}
tabs.forEach(t => t.addEventListener('click', () => showTab(t.dataset.tab)));

const state = {
  built: false,
  solved: false,
  board: [],          // card strings
  polling: null,
};

// ---------------------------------------------------------------------------
// Range editor
// ---------------------------------------------------------------------------

const editor = new RangeEditor($('range-matrix'), {
  textEl: $('range-text'),
  countEl: $('range-combo-count'),
  brushEl: $('brush-weight'),
  brushValEl: $('brush-weight-val'),
});

document.querySelectorAll('.rtab').forEach(b => {
  b.addEventListener('click', () => {
    document.querySelectorAll('.rtab').forEach(x => x.classList.remove('active'));
    b.classList.add('active');
    editor.setPlayer(+b.dataset.player);
  });
});
$('range-clear').addEventListener('click', () => editor.clear());
$('range-all').addEventListener('click', () => editor.fillAll());
$('range-apply-text').addEventListener('click', () => editor.applyText());

// presets
api.presets().then(presets => {
  const sel = $('preset-select');
  presets.forEach((p, i) => {
    const o = document.createElement('option');
    o.value = i; o.textContent = p.name;
    sel.appendChild(o);
  });
  sel.addEventListener('change', () => {
    if (sel.value === '') return;
    editor.setWeightsFromText(presets[+sel.value].range);
    sel.value = '';
  });
}).catch(() => {});

// default starting ranges so the app is usable immediately
editor.setWeightsFromText(
  '55-22,A8s-A2s,K9s-K6s,Q9s-Q6s,J9s-J7s,T8s+,97s+,86s+,75s+,64s+,54s,AJo-A8o,KTo-K9o,QTo+,JTo,T9o,98o'
).then(() => {
  editor.setPlayer(1);
  document.querySelectorAll('.rtab').forEach(x =>
    x.classList.toggle('active', x.dataset.player === '1'));
  return editor.setWeightsFromText(
    '22+,A2s+,K6s+,Q8s+,J8s+,T8s+,97s+,87s,76s,65s,A8o+,KTo+,QTo+,JTo');
}).then(() => {
  editor.setPlayer(0);
  document.querySelectorAll('.rtab').forEach(x =>
    x.classList.toggle('active', x.dataset.player === '0'));
});

// ---------------------------------------------------------------------------
// Preflop study setup — derives a heads-up postflop spot for the solver.
// `pf` always exists; a READY (closed, heads-up) line drives pot/stack and the
// position labels, while an empty/incomplete line leaves manual entry in charge.
// ---------------------------------------------------------------------------

let pf = preflop.freshState(+$('pf-stack').value || 100);

function pfDerived() { return preflop.derive(pf); }

function relabelRangeTabs(d) {
  const tabs = document.querySelectorAll('.rtab');
  if (!tabs.length) return;
  tabs[0].textContent = d ? `${d.oop} · OOP` : 'OOP';
  tabs[1].textContent = d ? `${d.ip} · IP` : 'IP';
}

function renderPreflop() {
  const lineEl = $('pf-line'), derEl = $('pf-derived');
  if (!lineEl) return;
  const chips = pf.line.map(a => {
    const txt = a.type === 'raise' ? `${a.pos} ${a.toBb}` : `${a.pos} ${a.type}`;
    return `<span class="pf-chip pf-${a.type}">${txt}</span>`;
  }).join('');
  const who = preflop.nextActor(pf);
  let actsHtml = '';
  if (who) {
    pf._acts = preflop.legalActions(pf);
    actsHtml = `<span class="pf-actor">${who}:</span>` + pf._acts.map((a, k) => {
      if (a.type === 'raise') {
        const min = Math.max(a.toBb, 2);
        return `<span class="pf-raisewrap"><button class="btn pf-act" data-act="${k}">${a.label} to</button>` +
          `<input class="pf-size" data-act="${k}" type="number" min="${min}" step="0.5" value="${a.toBb}"><span class="dim">bb</span></span>`;
      }
      return `<button class="btn pf-act" data-act="${k}">${a.label}</button>`;
    }).join('');
  } else {
    actsHtml = '<span class="pf-closed">✓ action closed</span>';
  }
  lineEl.innerHTML =
    `<div class="pf-chips">${chips || '<span class="dim">preflop:</span>'}</div>` +
    `<div class="pf-actions">${actsHtml}</div>`;
  lineEl.querySelectorAll('.pf-act').forEach(b => b.addEventListener('click', () => {
    const k = +b.dataset.act, a = pf._acts[k];
    if (a.type === 'raise') {
      const inp = lineEl.querySelector(`.pf-size[data-act="${k}"]`);
      const toBb = Math.max(2, +inp.value || a.toBb);
      pf.line.push({ pos: who, type: 'raise', toBb: Math.round(toBb * 2) / 2 });
    } else {
      pf.line.push({ pos: who, type: a.type });
    }
    renderPreflop();
  }));

  const d = pfDerived();
  if (d.ready) {
    derEl.className = 'pf-derived ok';
    derEl.innerHTML = `→ <b>${d.oop}</b> out of position vs <b>${d.ip}</b> in position · ` +
      `pot <b>${d.potBb}bb</b> · effective <b>${d.effStackBb}bb</b> · paste each player's range below`;
    $('cfg-pot').value = d.potBb;
    $('cfg-stack').value = d.effStackBb;
    relabelRangeTabs(d);
  } else {
    derEl.className = 'pf-derived dim';
    derEl.textContent = pf.line.length
      ? `· ${d.reason}`
      : '· no preflop line — pot & stack below are manual. Pick a scenario or build a line to study a position spot.';
    relabelRangeTabs(null);
  }
}

// scenarios dropdown
(() => {
  const sel = $('pf-preset');
  if (!sel) return;
  preflop.PRESETS.forEach((p, i) => {
    const o = document.createElement('option');
    o.value = i; o.textContent = p.name;
    sel.appendChild(o);
  });
  sel.addEventListener('change', () => {
    if (sel.value === '') return;
    const p = preflop.PRESETS[+sel.value];
    pf = { stackBb: p.stackBb, ante: 0, line: p.line.map(a => ({ ...a })) };
    $('pf-stack').value = p.stackBb;
    sel.value = '';
    renderPreflop();
  });
})();
$('pf-stack').addEventListener('input', () => {
  pf.stackBb = Math.max(2, +$('pf-stack').value || 100);
  renderPreflop();
});
$('pf-undo').addEventListener('click', () => { pf.line.pop(); renderPreflop(); });
$('pf-reset').addEventListener('click', () => {
  pf = preflop.freshState(+$('pf-stack').value || 100);
  renderPreflop();
});
renderPreflop();

// ---------------------------------------------------------------------------
// Board picker
// ---------------------------------------------------------------------------

function renderBoardInput() {
  const el = $('board-input-cards');
  el.innerHTML = '';
  state.board.forEach((cs, k) => {
    const chip = cardChip(cs);
    chip.dataset.tip = 'Click to remove this card from the board.';
    chip.addEventListener('click', () => {
      state.board.splice(k, 1);
      renderBoardInput(); renderDeckPicker();
    });
    el.appendChild(chip);
  });
  for (let k = state.board.length; k < 5; k++) {
    const ph = document.createElement('div');
    ph.className = 'placeholder';
    el.appendChild(ph);
  }
}

function renderDeckPicker() {
  const el = $('deck-picker');
  el.innerHTML = '';
  // one row per suit (s/h/d/c), one column per rank (A..2)
  for (const s of [3, 2, 1, 0]) {
    for (let r = 12; r >= 0; r--) {
      const cs = RANKS[r] + SUITS[s];
      const b = document.createElement('button');
      b.className = 'pick';
      b.innerHTML = `${RANKS[r]}<span class="suit-${SUITS[s]}">${SUIT_GLYPH[SUITS[s]]}</span>`;
      if (state.board.includes(cs)) b.classList.add('on', `cbg-${SUITS[s]}`);
      else if (state.board.length >= 5) b.classList.add('used');
      b.addEventListener('click', () => {
        const i = state.board.indexOf(cs);
        if (i >= 0) state.board.splice(i, 1);
        else if (state.board.length < 5) state.board.push(cs);
        renderBoardInput(); renderDeckPicker();
      });
      el.appendChild(b);
    }
  }
}
state.board = ['K' + 's', '7' + 'h', '2' + 'd'];
renderBoardInput();
renderDeckPicker();

// ---------------------------------------------------------------------------
// Sizes table
// ---------------------------------------------------------------------------

const SIZE_ROWS = [
  ['OOP bet', 'oop', 'bet'], ['OOP raise', 'oop', 'raise'], ['OOP donk', 'oop', 'donk'],
  ['IP bet', 'ip', 'bet'], ['IP raise', 'ip', 'raise'],
];
const SIZE_TIPS = {
  'OOP bet': 'Sizes OOP may bet when no one has bet yet this street (e.g. a c-bet after raising, or a lead after a check-through). % of pot, space-separated.',
  'OOP raise': 'Sizes OOP may raise when facing a bet. % of pot after a hypothetical call, or 2.5x = 2.5 times the bet faced.',
  'OOP donk': 'Sizes for leading INTO the previous street&#39;s aggressor (e.g. betting the turn after check-calling the flop). Leave empty for standard trees where OOP just checks to the raiser.',
  'IP bet': 'Sizes IP may bet when checked to (or betting first after a check-through). % of pot, space-separated.',
  'IP raise': 'Sizes IP may raise when facing a bet or donk. % of pot after a call, or 2.5x multiples.',
};
const DEFAULT_SIZES = {
  oop: { bet: ['33', '75', '75'], raise: ['60', '60', '60'], donk: ['', '', ''] },
  ip: { bet: ['33 75', '75', '75'], raise: ['60', '60', '60'], donk: ['', '', ''] },
};

function buildSizesTable() {
  const tbody = $('sizes-body');
  tbody.innerHTML = '';
  for (const [label, who, kind] of SIZE_ROWS) {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td data-tip="${SIZE_TIPS[label]}">${label}</td>` + [0, 1, 2].map(st =>
      `<td><input type="text" data-who="${who}" data-kind="${kind}" data-street="${st}"
        value="${DEFAULT_SIZES[who][kind][st]}"
        ${kind === 'donk' && st === 0 ? 'disabled placeholder="—"' : ''}></td>`).join('');
    tbody.appendChild(tr);
  }
}
buildSizesTable();

function collectSizes(who) {
  const out = [];
  for (let st = 0; st < 3; st++) {
    const get = kind => {
      const input = document.querySelector(
        `#sizes-body input[data-who="${who}"][data-kind="${kind}"][data-street="${st}"]`);
      return input && !input.disabled ? input.value.trim() : '';
    };
    out.push({ bet: get('bet'), raise: get('raise'), donk: who === 'oop' ? get('donk') : '' });
  }
  return out;
}

// ---------------------------------------------------------------------------
// Build tree
// ---------------------------------------------------------------------------

// ---- memory / GPU-eligibility readout -------------------------------------
function gpuPlanText(info) {
  if (info.gpu_available === undefined) return ''; // server too old to report it
  if (!info.gpu_available) return 'GPU off — solves on CPU';
  const budget = (info.gpu_cap_mb / 1000).toFixed(0);
  if (!info.vram_mb || info.vram_mb <= info.gpu_cap_mb) return `fits GPU ✓ (~${budget} GB free)`;
  return `exceeds GPU ✗ — needs ${(info.vram_mb / 1000).toFixed(1)} GB > ~${budget} GB free → CPU`;
}
function memSummary(info) {
  const ram = `RAM ~${(info.arena_mb / 1000).toFixed(2)} GB`;
  const vram = info.vram_mb ? ` · VRAM ~${(info.vram_mb / 1000).toFixed(2)} GB` : '';
  const plan = gpuPlanText(info);
  return `${ram}${vram}${plan ? ' · ' + plan : ''}`;
}
// live compute indicator from a poll status. Only asserts GPU/CPU once a solve
// has actually started this session; before that the plan lives in #mem-info.
function computeText(st) {
  if (!st.tree) return '';
  if (st.gpu) return st.state === 'running' ? '⚡ computing on GPU' : '⚡ solved on GPU';
  if (st.gpu_note) return `△ ${st.gpu_note}`;
  if (st.state === 'running') return st.tree.gpu_available ? 'starting GPU…' : 'computing on CPU';
  return ''; // built/loaded, not solved yet — #mem-info shows whether it fits GPU
}

$('btn-build').addEventListener('click', async () => {
  if (state.board.length < 3) return toast('pick at least a 3-card flop', true);
  const cfg = {
    board: state.board.join(''),
    range_oop: editor.textFor(0),
    range_ip: editor.textFor(1),
    starting_pot: +$('cfg-pot').value,
    effective_stack: +$('cfg-stack').value,
    rake_pct: +$('cfg-rake').value,
    rake_cap: +$('cfg-rakecap').value,
    allin_threshold: +$('cfg-allinthr').value,
    add_allin: $('cfg-addallin').checked,
    max_raises: 10,
    oop: collectSizes('oop'),
    ip: collectSizes('ip'),
  };
  if (!cfg.range_oop) return toast('OOP range is empty', true);
  if (!cfg.range_ip) return toast('IP range is empty', true);
  $('btn-build').disabled = true;
  $('build-info').textContent = 'building…';
  try {
    const info = await api.buildSpot(cfg);
    state.built = true;
    state.solved = false;
    browser.reset(); // new tree: drop any stale browse line
    // Hand the preflop context to Browse for position labels + a preflop ribbon
    // (null when the spot was set up manually with no preflop line).
    const pd = pfDerived();
    browser.preflop = pd.ready
      ? { oop: pd.oop, ip: pd.ip, potBb: pd.potBb, effStackBb: pd.effStackBb,
          segments: preflop.preflopSegments(pf) }
      : null;
    localStorage.setItem('freepio-last-spot', JSON.stringify(cfg));
    const summary = `${info.nodes.toLocaleString()} nodes · ${info.action_nodes.toLocaleString()} decision points · ` +
      `hands ${info.hands_oop}/${info.hands_ip}`;
    $('build-info').textContent = `${summary} · ${memSummary(info)}`;
    $('tree-summary').textContent = `board ${info.board} · ${summary}`;
    $('mem-info').textContent = memSummary(info);
    $('compute-info').textContent = '';
    toast('tree built — go to SOLVE');
    showTab('solve');
  } catch (e) {
    $('build-info').textContent = '';
    toast(`build failed: ${e.message}`, true);
  } finally {
    $('btn-build').disabled = false;
  }
});

// ---------------------------------------------------------------------------
// Solve dashboard
// ---------------------------------------------------------------------------

$('btn-solve').addEventListener('click', async () => {
  if (!state.built) return toast('build a tree first', true);
  try {
    await api.solve({
      max_iterations: +$('run-iters').value,
      target_exploit_pct: +$('run-target').value,
      check_every: +$('run-check').value,
    });
    toast('solving…');
    startPolling();
  } catch (e) {
    toast(e.message, true);
  }
});

$('btn-stop').addEventListener('click', async () => {
  await api.stop().catch(() => {});
});

// Header solve controls — solve/stop/monitor from any screen, so Browse is the
// operational home and re-solving (e.g. after a node lock) never needs a tab switch.
$('btn-solve-top').addEventListener('click', () => $('btn-solve').click());
$('btn-stop-top').addEventListener('click', () => $('btn-stop').click());
let firstPoll = true;

function startPolling() {
  if (state.polling) clearInterval(state.polling);
  state.polling = setInterval(pollStatus, 1000);
  pollStatus();
}

async function pollStatus() {
  let st;
  try { st = await api.status(); } catch { return; }
  const pill = $('status-pill');
  pill.className = '';
  if (st.state === 'running') pill.classList.add('running');
  else if (st.state === 'done') pill.classList.add('done');
  else if (st.state === 'ready') pill.classList.add('ready');
  $('status-text').textContent = st.state + (st.state === 'running' ? ` · iter ${st.iteration}` : '');

  $('iter-now').textContent = st.iteration;
  $('elapsed-now').textContent = `${Math.round(st.elapsed_secs)}s`;
  $('exploit-now').textContent = st.exploit_pct > 0 ? st.exploit_pct.toFixed(3) : '—';
  drawChart(st.history || []);

  if (st.tree && !state.built) {
    state.built = true;
    $('tree-summary').textContent = `board ${st.tree.board} · ${st.tree.nodes.toLocaleString()} nodes · ${(st.tree.arena_mb/1000).toFixed(2)} GB`;
  }
  if (st.tree) $('mem-info').textContent = memSummary(st.tree);
  const ci = $('compute-info');
  ci.textContent = computeText(st);
  ci.className = 'mono' + (st.gpu ? ' gpu-on' : (st.gpu_note ? ' gpu-fallback' : ' dim'));

  // header solve bar: present once a tree exists; solve from anywhere
  const built = !!st.tree, running = st.state === 'running';
  $('solve-controls').classList.toggle('hidden', !built);
  $('btn-solve-top').classList.toggle('hidden', running);
  $('btn-stop-top').classList.toggle('hidden', !running);
  $('btn-solve-top').textContent = (st.state === 'done' || st.state === 'stopped') ? 'RE-SOLVE' : 'SOLVE';
  $('solve-readout').textContent = built
    ? `iter ${st.iteration}${st.exploit_pct > 0 ? ` · ${st.exploit_pct.toFixed(2)}% pot` : ''}` : '';
  // land on Browse when there's already a solved spot to study (first poll only)
  if (firstPoll) {
    firstPoll = false;
    if (st.state === 'done' && st.tree) showTab('browse');
  }
  if (st.state === 'done' || st.state === 'stopped') {
    if (!state.solved && st.iteration > 0) {
      state.solved = true;
      toast(`solve ${st.state} — exploitability ${st.exploit_pct.toFixed(3)}% pot`);
    }
    if (state.polling && st.state !== 'running') {
      clearInterval(state.polling);
      state.polling = null;
    }
  }
}

function drawChart(history) {
  const cv = $('convergence-chart');
  const ctx = cv.getContext('2d');
  const W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  ctx.font = '11px IBM Plex Mono';
  if (history.length < 2) {
    ctx.fillStyle = '#4d5564';
    ctx.fillText('convergence chart appears here during solving', 20, H / 2);
    return;
  }
  const target = +$('run-target').value || 0.3;
  const xs = history.map(h => h.iteration);
  const ys = history.map(h => Math.max(h.exploit_pct, 1e-3));
  const xMax = Math.max(...xs);
  const yMax = Math.max(...ys, target * 2);
  const yMin = Math.min(...ys, target / 2);
  const lY = v => {
    const t = (Math.log10(v) - Math.log10(yMin)) / (Math.log10(yMax) - Math.log10(yMin) + 1e-12);
    return H - 28 - t * (H - 48);
  };
  const lX = v => 46 + (v / xMax) * (W - 66);

  // grid lines at decades
  ctx.strokeStyle = '#262b34';
  ctx.fillStyle = '#5b6474';
  for (let d = Math.ceil(Math.log10(yMin)); d <= Math.floor(Math.log10(yMax)); d++) {
    const v = Math.pow(10, d);
    ctx.beginPath(); ctx.moveTo(46, lY(v)); ctx.lineTo(W - 20, lY(v)); ctx.stroke();
    ctx.fillText(`${v}%`, 8, lY(v) + 4);
  }
  // target line
  ctx.strokeStyle = '#3d86c6'; ctx.setLineDash([5, 4]);
  ctx.beginPath(); ctx.moveTo(46, lY(target)); ctx.lineTo(W - 20, lY(target)); ctx.stroke();
  ctx.setLineDash([]);
  // curve
  ctx.strokeStyle = '#2fc26e'; ctx.lineWidth = 2;
  ctx.beginPath();
  history.forEach((h, k) => {
    const x = lX(h.iteration), y = lY(Math.max(h.exploit_pct, 1e-3));
    if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.lineWidth = 1;
  ctx.fillStyle = '#7f8898';
  ctx.fillText(`iter ${xMax}`, W - 80, H - 8);
}

// ---------------------------------------------------------------------------
// Save / load
// ---------------------------------------------------------------------------

async function refreshSaves() {
  try {
    const names = await api.saves();
    const sel = $('load-select');
    sel.innerHTML = '<option value="">saved solves…</option>';
    names.forEach(n => {
      const o = document.createElement('option');
      o.value = n; o.textContent = n;
      sel.appendChild(o);
    });
  } catch {}
}
refreshSaves();

$('btn-save').addEventListener('click', async () => {
  const name = $('save-name').value.trim();
  if (!name) return toast('enter a save name', true);
  try {
    await api.save(name);
    toast('saved');
    refreshSaves();
  } catch (e) { toast(e.message, true); }
});

$('btn-load').addEventListener('click', async () => {
  const name = $('load-select').value;
  if (!name) return toast('pick a save', true);
  try {
    await api.load(name);
    state.built = true; state.solved = true;
    browser.reset(); // different solve: drop any stale browse line
    toast('loaded — go to BROWSE');
    pollStatus();
  } catch (e) { toast(e.message, true); }
});

// ---------------------------------------------------------------------------
// Browser
// ---------------------------------------------------------------------------

const browser = new Browser({
  matrix: $('browse-matrix'),
  legend: $('matrix-legend'),
  history: $('history-bar'),
  histLeft: $('hist-left'),
  histRight: $('hist-right'),
  pot: $('browse-pot'),
  actionList: $('action-list'),
  actionsTitle: $('actions-title'),
  cardPicker: $('card-picker'),
  runouts: $('runouts-report'),
  handsContent: $('hands-content'),
  handsLabel: $('hands-label'),
  handsTabs: document.querySelector('.hands-tabs'),
  eqCanvas: $('equity-chart'),
  eqStats: $('eqchart-stats'),
  segPlayer: $('seg-player'),
  segMode: $('seg-mode'),
});
// expose for console debugging / tooling
window.browser = browser;

$('seg-player').querySelectorAll('button').forEach(b =>
  b.addEventListener('click', () => {
    browser.player = +b.dataset.v;
    browser.syncSegs(); browser.renderActions(); browser.renderMatrix(); browser.renderLegend();
    browser.renderHandsPanel(); browser.drawEquityChart();
  }));
$('seg-mode').querySelectorAll('button').forEach(b =>
  b.addEventListener('click', () => {
    browser.mode = b.dataset.v;
    // renderActions too: the EXPLOIT banner lives in the actions panel and
    // must appear/disappear with the mode
    browser.syncSegs(); browser.renderActions(); browser.renderMatrix(); browser.renderLegend();
  }));

// cell-display menu: fill mode (Normalized/Range/Full) + orientation (Vertical/Horizontal)
const viewMenu = $('view-menu');
const viewMenuBtn = $('view-menu-btn');
function syncViewMenu() {
  viewMenu.querySelectorAll('[data-fill]').forEach(b =>
    b.classList.toggle('sel', b.dataset.fill === browser.fillMode));
  viewMenu.querySelectorAll('[data-orient]').forEach(b =>
    b.classList.toggle('sel', (b.dataset.orient === 'vertical') === browser.comboRows));
}
viewMenuBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  const open = viewMenu.classList.toggle('hidden') === false;
  viewMenuBtn.classList.toggle('open', open);
  if (open) syncViewMenu();
});
viewMenu.querySelectorAll('.vm-item').forEach(b =>
  b.addEventListener('click', (e) => {
    e.stopPropagation();
    if (b.dataset.fill) browser.fillMode = b.dataset.fill;
    else browser.comboRows = b.dataset.orient === 'vertical';
    syncViewMenu();
    browser.renderMatrix();
  }));
document.addEventListener('click', (e) => {
  if (!viewMenu.classList.contains('hidden') && !e.target.closest('.view-menu-wrap')) {
    viewMenu.classList.add('hidden');
    viewMenuBtn.classList.remove('open');
  }
});
syncViewMenu();

// restore last config if present
try {
  const saved = JSON.parse(localStorage.getItem('freepio-last-spot') || 'null');
  if (saved) {
    $('cfg-pot').value = saved.starting_pot;
    $('cfg-stack').value = saved.effective_stack;
    $('cfg-rake').value = saved.rake_pct;
    $('cfg-rakecap').value = saved.rake_cap;
    $('cfg-allinthr').value = saved.allin_threshold;
    $('cfg-addallin').checked = saved.add_allin;
    if (saved.board) {
      state.board = saved.board.match(/.{2}/g) || state.board;
      renderBoardInput(); renderDeckPicker();
    }
    for (const who of ['oop', 'ip']) {
      (saved[who] || []).forEach((s, st) => {
        for (const kind of ['bet', 'raise', 'donk']) {
          const input = document.querySelector(
            `#sizes-body input[data-who="${who}"][data-kind="${kind}"][data-street="${st}"]`);
          if (input && s[kind] !== undefined) input.value = s[kind];
        }
      });
    }
  }
} catch {}

// initial status poll (e.g. after page reload during a solve)
pollStatus();
startPolling();
