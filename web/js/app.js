// App bootstrap: tabs, setup form, solve dashboard, browser wiring.

import { api, toast } from './api.js';
import { RangeEditor } from './range_editor.js';
import { Browser, cardChip } from './browse.js';
import { RANKS, SUITS, SUIT_GLYPH, cardToString } from './cards.js';
import { initTooltips } from './tooltip.js';

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
    localStorage.setItem('freepio-last-spot', JSON.stringify(cfg));
    const summary = `${info.nodes.toLocaleString()} nodes · ${info.action_nodes.toLocaleString()} decision points · ` +
      `${(info.arena_mb / 1000).toFixed(2)} GB solver memory · hands ${info.hands_oop}/${info.hands_ip}`;
    $('build-info').textContent = summary;
    $('tree-summary').textContent = `board ${info.board} · ${summary}`;
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

$('seg-player').querySelectorAll('button').forEach(b =>
  b.addEventListener('click', () => {
    browser.player = +b.dataset.v;
    browser.syncSegs(); browser.renderMatrix(); browser.renderLegend();
    browser.renderHandsPanel(); browser.drawEquityChart();
  }));
$('seg-mode').querySelectorAll('button').forEach(b =>
  b.addEventListener('click', () => {
    browser.mode = b.dataset.v;
    browser.syncSegs(); browser.renderMatrix(); browser.renderLegend();
  }));

$('combo-toggle').addEventListener('click', () => {
  browser.comboRows = !browser.comboRows;
  $('combo-toggle').classList.toggle('active', browser.comboRows);
  browser.renderMatrix();
});

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
