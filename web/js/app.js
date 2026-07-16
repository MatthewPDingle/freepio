// App bootstrap: tabs, setup form, solve dashboard, browser wiring.

import { api, toast } from './api.js';
import { RangeEditor } from './range_editor.js';
import { Browser, cardChip } from './browse.js';
import { RANKS, SUITS, SUIT_GLYPH, cardToString } from './cards.js';
import { initTooltips } from './tooltip.js';
import { initPreflopLab } from './preflop_lab.js';
import { initReports } from './reports.js';

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
  // gate on built, not solved: a built tree is browsable even mid-solve (the
  // server keeps mid-solve browse data fresh) — gating on solved showed the
  // PREVIOUS tree's solution until the first solve finished
  if (name === 'browse' && state.built) browser.refresh();
  if (name === 'solve') drawChart(chartHistory); // canvas can't size/draw while hidden
}
tabs.forEach(t => t.addEventListener('click', () => showTab(t.dataset.tab)));

const state = {
  built: false,
  solved: false,
  saved: false,         // current solve persisted via save/load (gates discard confirms)
  pendingPreflop: null, // last PREFLOP LAB export applied to SETUP
  pendingPreflopKey: null, // SETUP fingerprint of that export (staleness check)
  treeStamp: null,      // board:nodes of the tree behind #tree-summary

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

// Range presets are Preflop Lab exports: each one restores both ranges,
// pot, stack and rake in one click (SEND TO POSTFLOP creates them).
// one-time rebrand migration: carry saved lab spots / last config over
for (const [o, n] of [['freepio-pfl-spots', 'gtopen-pfl-spots'],
                      ['freepio-last-spot', 'gtopen-last-spot']]) {
  if (localStorage.getItem(o) != null && localStorage.getItem(n) == null) {
    localStorage.setItem(n, localStorage.getItem(o));
  }
}
const pflSpots = () => JSON.parse(localStorage.getItem('gtopen-pfl-spots') || '[]');

function refreshPresetDropdown() {
  const sel = $('preset-select');
  sel.innerHTML = '<option value="">from Preflop Lab\u2026</option>';
  const labs = pflSpots();
  if (!labs.length) {
    const o = document.createElement('option');
    o.disabled = true;
    o.textContent = 'nothing yet \u2014 SEND TO POSTFLOP in the Preflop Lab saves spots here';
    sel.appendChild(o);
    return;
  }
  labs.forEach((sp, i) => {
    const o = document.createElement('option');
    o.value = `pfl:${i}`; o.textContent = sp.name;
    sel.appendChild(o);
  });
}
refreshPresetDropdown();
$('preset-select').addEventListener('change', async () => {
  const sel = $('preset-select');
  const v = sel.value;
  sel.value = '';
  if (!v.startsWith('pfl:')) return;
  const sp = pflSpots()[+v.slice(4)];
  if (sp) await applyExportedSpot(sp);
});

/** Fingerprint of the SETUP fields a Preflop Lab export pins down (ranges,
 *  pot, stack) — compared at build time so a build that no longer matches the
 *  export doesn't inherit its ribbon/positions/villain profiles. */
function preflopKey() {
  return JSON.stringify([editor.textFor(0), editor.textFor(1),
    +$('cfg-pot').value, +$('cfg-stack').value]);
}

/** Load a preflop-lab export into SETUP: both ranges, pot, stack, tab labels.
 *  Also remembered so BUILD TREE hands the preflop line to Browse's ribbon. */
async function applyExportedSpot(ex) {
  $('cfg-pot').value = ex.pot_bb;
  $('cfg-stack').value = ex.eff_stack_bb;
  if (ex.rake_pct != null) $('cfg-rake').value = ex.rake_pct;
  if (ex.rake_cap != null) $('cfg-rakecap').value = ex.rake_cap;
  editor.setPlayer(1);
  await editor.setWeightsFromText(ex.range_ip);
  editor.setPlayer(0);
  await editor.setWeightsFromText(ex.range_oop);
  state.pendingPreflop = ex;
  state.pendingPreflopKey = preflopKey();
  // the range tabs carry the positions from the lab: "BB · OOP" / "UTG · IP"
  const rtabs = document.querySelectorAll('.rtab');
  rtabs.forEach((x, i) => x.classList.toggle('active', i === 0));
  if (rtabs.length >= 2) {
    rtabs[0].textContent = `${ex.oop_pos} \u00b7 OOP`;
    rtabs[1].textContent = `${ex.ip_pos} \u00b7 IP`;
  }
  showTab('setup');
}

// last-built spot from a previous session (the rest of it is restored into
// SETUP at the bottom of this file)
let lastSpot = null;
try { lastSpot = JSON.parse(localStorage.getItem('gtopen-last-spot') || 'null'); } catch {}

// starting ranges: the last-built spot's when present — a rebuild after a
// reload must solve what the user configured, not the demo — else demo
// defaults so the app is usable immediately
editor.setWeightsFromText(lastSpot && lastSpot.range_oop ? lastSpot.range_oop :
  '55-22,A8s-A2s,K9s-K6s,Q9s-Q6s,J9s-J7s,T8s+,97s+,86s+,75s+,64s+,54s,AJo-A8o,KTo-K9o,QTo+,JTo,T9o,98o'
).then(() => {
  editor.setPlayer(1);
  document.querySelectorAll('.rtab').forEach(x =>
    x.classList.toggle('active', x.dataset.player === '1'));
  return editor.setWeightsFromText(lastSpot && lastSpot.range_ip ? lastSpot.range_ip :
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
    tr.innerHTML = `<td data-tip="${SIZE_TIPS[label]}" tabindex="0">${label}</td>` + [0, 1, 2].map(st =>
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

/** The SETUP page's current spot as a SpotRequest (shared by BUILD and
 *  flop reports). Returns null when the config is incomplete. */
function currentSpotRequest() {
  if (state.board.length < 3) return null;
  return {
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
}

/** Build the tree from SETUP. Resolves true when the build landed — shared by
 *  the BUILD TREE button and the reports' OPEN IN BROWSE build-then-solve. */
async function buildTree() {
  if (state.board.length < 3) { toast('pick at least a 3-card flop', true); return false; }
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
  if (!cfg.range_oop) { toast('OOP range is empty', true); return false; }
  if (!cfg.range_ip) { toast('IP range is empty', true); return false; }
  if (state.solved && !state.saved &&
      !confirm('The current solve has not been saved — building a new tree discards it. Continue?')) {
    return false;
  }
  $('btn-build').disabled = true;
  const prevBuildInfo = $('build-info').textContent;
  $('build-info').textContent = 'building…';
  // the server validates the new spot BEFORE swapping sessions, so the old
  // tree (and any solve on it) survives until the build actually lands. Reset
  // the client's claims optimistically for the in-flight build, but keep the
  // old values to restore if the build is refused.
  const prevBuilt = state.built;
  const prevSolved = state.solved;
  state.built = false;
  state.solved = false;
  try {
    const info = await api.buildSpot(cfg);
    state.built = true;
    browser.reset(); // new tree: drop any stale browse line
    // Preflop context for Browse (position labels + ribbon prefix) comes from
    // the last Preflop Lab export — but only while SETUP still matches it. A
    // build with edited ranges/pot/stack is a different spot: drop the stale
    // lab context (ribbon, positions, villain profiles) for good.
    if (state.pendingPreflop && preflopKey() !== state.pendingPreflopKey) {
      state.pendingPreflop = null;
      state.pendingPreflopKey = null;
      const rtabs = document.querySelectorAll('.rtab');
      if (rtabs.length >= 2) { rtabs[0].textContent = 'OOP'; rtabs[1].textContent = 'IP'; }
    }
    if (state.pendingPreflop) {
      const ex = state.pendingPreflop;
      browser.preflop = { oop: ex.oop_pos, ip: ex.ip_pos, potBb: ex.pot_bb,
        effStackBb: ex.eff_stack_bb, segments: ex.segments || [],
        villains: ex.villains || null, aggressor: ex.aggressor ?? null };
    } else {
      browser.preflop = null;
    }
    localStorage.setItem('gtopen-last-spot', JSON.stringify(cfg));
    const summary = `${info.nodes.toLocaleString()} nodes · ${info.action_nodes.toLocaleString()} decision points · ` +
      `hands ${info.hands_oop}/${info.hands_ip}`;
    $('build-info').textContent = `${summary} · ${memSummary(info)}`;
    $('tree-summary').textContent = `board ${info.board} · ${summary}`;
    state.treeStamp = `${info.board}:${info.nodes}`; // pollStatus keeps this text for the same tree
    $('mem-info').textContent = memSummary(info);
    $('compute-info').textContent = '';
    toast('tree built — go to SOLVE');
    showTab('solve');
    return true;
  } catch (e) {
    // validate-then-swap on the server: a refused build leaves the previous
    // session — tree, and any completed solve — fully intact. Restore the
    // client's claims to match (state.treeStamp and the tree/mem displays
    // still describe that surviving session, so they stay). Restoring
    // `solved` BEFORE pollStatus is what keeps the completion branch there
    // from mistaking the surviving done session for a fresh solve and
    // toasting 'solve done' over this error toast (plus a needless Browse
    // refresh) — the toast must fire only when a run actually finishes.
    state.built = prevBuilt;
    state.solved = prevSolved;
    $('build-info').textContent = prevBuildInfo;
    toast(`build failed: ${e.message}`, true);
    pollStatus(); // resync pill + header solve bar with the surviving session
    return false;
  } finally {
    $('btn-build').disabled = false;
  }
}
$('btn-build').addEventListener('click', () => { buildTree(); });

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
    // a fresh run: completion (toast + Browse refresh below) keys on this
    // transition, and the new strategies aren't on disk yet
    state.solved = false;
    state.saved = false;
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

// iterations/sec from successive status polls (client-side delta, so it
// survives resumed solves where iteration is cumulative but elapsed resets);
// lightly smoothed to keep the readout steady.
let ipsPrev = null, ipsEma = null;
function updateIps(st) {
  const el = $('ips-now');
  if (!el) return;
  if (st.state !== 'running') { ipsPrev = null; ipsEma = null; return; }
  const now = performance.now();
  if (ipsPrev && st.iteration > ipsPrev.iter) {
    const inst = (st.iteration - ipsPrev.iter) / ((now - ipsPrev.t) / 1000);
    ipsEma = ipsEma == null ? inst : 0.7 * ipsEma + 0.3 * inst;
    el.textContent = ipsEma >= 10 ? ipsEma.toFixed(0) : ipsEma.toFixed(1);
  }
  if (!ipsPrev || st.iteration !== ipsPrev.iter) ipsPrev = { iter: st.iteration, t: now };
}

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
  updateIps(st);
  $('exploit-now').textContent = st.exploit_pct > 0 ? st.exploit_pct.toFixed(3) : '—';
  drawChart(st.history || []);

  // keep the SOLVE tab's tree line in sync with the server session — load
  // (and any other session swap) must not keep showing the previous build.
  // The stamp keeps BUILD's richer summary in place for the same tree.
  if (st.tree) {
    state.built = true;
    const stamp = `${st.tree.board}:${st.tree.nodes}`;
    if (state.treeStamp !== stamp) {
      state.treeStamp = stamp;
      $('tree-summary').textContent =
        `board ${st.tree.board} · ${st.tree.nodes.toLocaleString()} nodes · ${(st.tree.arena_mb/1000).toFixed(2)} GB`;
    }
    $('mem-info').textContent = memSummary(st.tree);
  } else {
    state.built = false;
  }
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
  // solve completion: toast once per run, and refresh an open Browse view so
  // the lock → re-solve → inspect loop shows the adapted strategies (the view
  // renders from a cached node payload until refreshed)
  if (st.state === 'done' || st.state === 'stopped') {
    if (!state.solved && st.iteration > 0) {
      state.solved = true;
      toast(`solve ${st.state} — exploitability ${st.exploit_pct.toFixed(3)}% pot`);
      if ($('view-browse').classList.contains('active')) browser.refresh();
    }
    if (state.polling && st.state !== 'running') {
      clearInterval(state.polling);
      state.polling = null;
    }
  }
  // land on Browse when there's already a solved spot to study (first poll
  // only, AFTER state.solved is settled above so the tab switch renders)
  if (firstPoll) {
    firstPoll = false;
    if (st.tree) {
      // Deep link: open Browse at a node, e.g. /#line=a1,a1,cQh
      // (a<i> = action index i, c<card> = turn/river card). Honored whenever
      // a tree exists — done, stopped and mid-solve are all browsable.
      const m = location.hash.match(/^#line=([a-zA-Z0-9,]+)$/);
      if (m) {
        // set the path first: the tab switch does the (single) node fetch,
        // and a malformed link falls back to the root via refresh()'s recovery
        browser.path = m[1].split(',').map(t =>
          t[0] === 'a' ? { type: 'action', index: +t.slice(1) }
                       : { type: 'card', card: t.slice(1) });
        showTab('browse');
      } else if (st.state === 'done') {
        showTab('browse');
      }
    }
  }
}

let chartHistory = []; // last drawn data, for redraws on tab switch / resize
function drawChart(history) {
  chartHistory = history;
  const cv = $('convergence-chart');
  // render at the element's CSS size x devicePixelRatio so text stays crisp
  // (the CSS stretches the canvas to width:100%); skip while the SOLVE tab is
  // hidden (zero rect) — showTab redraws on entry
  const rect = cv.getBoundingClientRect();
  if (rect.width < 10) return;
  const dpr = window.devicePixelRatio || 1;
  const wantW = Math.round(rect.width * dpr);
  const wantH = Math.round(rect.height * dpr);
  if (cv.width !== wantW || cv.height !== wantH) {
    cv.width = wantW;
    cv.height = wantH;
  }
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const W = rect.width, H = rect.height;
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
window.addEventListener('resize', () => drawChart(chartHistory));

// ---------------------------------------------------------------------------
// Save / load
// ---------------------------------------------------------------------------

let saveNames = []; // known names, for the overwrite confirm
async function refreshSaves() {
  try {
    const names = await api.saves();
    saveNames = names;
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
  if (saveNames.includes(name) &&
      !confirm(`"${name}" already exists — overwrite it?`)) return;
  try {
    await api.save(name);
    state.saved = true;
    toast('saved');
    refreshSaves();
  } catch (e) { toast(e.message, true); }
});

$('btn-load').addEventListener('click', async () => {
  const name = $('load-select').value;
  if (!name) return toast('pick a save', true);
  if (state.solved && !state.saved &&
      !confirm('The current solve has not been saved — loading discards it. Continue?')) return;
  try {
    await api.load(name);
    state.built = true; state.solved = true;
    state.saved = true; // it IS the on-disk copy
    browser.reset(); // different solve: drop any stale browse line
    toast('loaded — go to BROWSE');
    pollStatus(); // also refreshes the SOLVE tab's tree line from the new session
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

// ---------------------------------------------------------------------------
// Preflop lab (multiway preflop solver) + its bridge into SETUP
// ---------------------------------------------------------------------------

initPreflopLab({
  els: {
    preset: $('pfl-preset'), players: $('pfl-players'), stack: $('pfl-stack'),
    opens: $('pfl-opens'), mult: $('pfl-mult'), maxRaises: $('pfl-maxraises'),
    ante: $('pfl-ante'), rakePct: $('pfl-rakepct'), rakeCap: $('pfl-rakecap'),
    limp: $('pfl-limp'), allin: $('pfl-allin'), realization: $('pfl-realization'),
    build: $('pfl-build'), solve: $('pfl-solve'), stop: $('pfl-stopbtn'),
    prog: $('pfl-prog'),
    buildInfo: $('pfl-buildinfo'), status: $('pfl-status'),
    ribbon: $('pfl-ribbon'), nodeTitle: $('pfl-nodetitle'),
    seats: $('pfl-seats'), rangeSeg: $('pfl-rangeseg'), grid: $('pfl-grid'),
    legend: $('pfl-legend'), gridCap: $('pfl-gridcap'), exportBtn: $('pfl-export'),
    fillSeg: $('pfl-fillseg'), estimate: $('pfl-estimate'),
    modelBox: $('pfl-model'), editor: $('pfl-editor'),
    hero: $('pfl-hero'), applyBtn: $('pfl-apply'),
    savedSel: $('pfl-savedsel'), saveGame: $('pfl-savegame'),
  },
  toast,
  gotoSetup: () => showTab('setup'),
  onExport: async (ex, lineText) => {
    ex.name = `${ex.oop_pos} vs ${ex.ip_pos} \u00b7 ${ex.pot_bb}bb pot \u00b7 ${lineText}`;
    const spots = pflSpots().filter(sp => sp.name !== ex.name);
    spots.unshift(ex);
    localStorage.setItem('gtopen-pfl-spots', JSON.stringify(spots.slice(0, 20)));
    refreshPresetDropdown();
    await applyExportedSpot(ex);
  },
});

// ---------------------------------------------------------------------------
// Flop reports
// ---------------------------------------------------------------------------

/** Apply a report's spot + a chosen board into SETUP, then build and solve. */
async function openReportSpot(spot, board) {
  $('cfg-pot').value = spot.starting_pot;
  $('cfg-stack').value = spot.effective_stack;
  $('cfg-rake').value = spot.rake_pct;
  $('cfg-rakecap').value = spot.rake_cap;
  $('cfg-allinthr').value = spot.allin_threshold;
  $('cfg-addallin').checked = spot.add_allin;
  state.board = board.match(/.{2}/g) || [];
  renderBoardInput(); renderDeckPicker();
  for (const who of ['oop', 'ip']) {
    (spot[who] || []).forEach((sz, st) => {
      for (const kind of ['bet', 'raise', 'donk']) {
        const input = document.querySelector(
          `#sizes-body input[data-who="${who}"][data-kind="${kind}"][data-street="${st}"]`);
        if (input && sz[kind] != null) input.value = sz[kind];
      }
    });
  }
  editor.setPlayer(1);
  await editor.setWeightsFromText(spot.range_ip);
  editor.setPlayer(0);
  await editor.setWeightsFromText(spot.range_oop);
  showTab('setup');
  toast(`report spot loaded on ${board} — building…`);
  // await the build directly (polling a client flag raced against in-flight
  // builds), then solve, then land in Browse
  if (!await buildTree()) return;
  $('btn-solve').click();
  showTab('browse');
}

const reports = initReports({
  els: {
    name: $('rep-name'), flops: $('rep-flops'), vsVillain: $('rep-vsvillain'),
    run: $('rep-run'), stop: $('rep-stop'), progress: $('rep-progress'),
    library: $('rep-library'), viewer: $('rep-viewer'), title: $('rep-title'),
    subtitle: $('rep-subtitle'), controls: $('rep-controls'),
    canvas: $('rep-canvas'), detail: $('rep-detail'),
    aggregate: $('rep-aggregate'), table: $('rep-table'), legend: $('rep-legend'),
  },
  toast,
  currentSpot: currentSpotRequest,
  villains: () => {
    const pf = browser.preflop;
    if (!pf || !pf.villains) return null;
    for (const [side, p] of [['oop', 0], ['ip', 1]]) {
      const v = pf.villains[side];
      if (v) return { player: p, name: v.name, stats: v.stats, aggressor: pf.aggressor };
    }
    return null;
  },
  openInBrowse: openReportSpot,
});
void reports;

// restore last config if present (the ranges were restored with the editor
// bootstrap above, so a rebuild really reproduces the saved spot)
try {
  const saved = lastSpot;
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
