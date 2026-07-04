// PREFLOP LAB — UI for the multiway preflop solver (equity-model postflop).
// Build a game (any limps/sizes/players), solve it, walk the action tree with
// a GTO Wizard-style ribbon, and export any heads-up flop node straight into
// the postflop solver's SETUP.

import { api } from './api.js';
import { cellInfo } from './cards.js';

const COLORS = { fold: '#4a78c8', check: '#5ca75f', call: '#5ca75f' };
const RAISE_SHADES = ['#e8484c', '#c73e55', '#a4335f', '#7d3ca3'];

function positionsFor(nPlayers) {
  const NAMED = {
    2: ['SB', 'BB'], // HU: SB is the button, acts first pre
    3: ['BTN', 'SB', 'BB'],
    6: ['UTG', 'HJ', 'CO', 'BTN', 'SB', 'BB'],
    8: ['UTG', 'UTG1', 'MP', 'HJ', 'CO', 'BTN', 'SB', 'BB'],
    9: ['UTG', 'UTG1', 'MP', 'LJ', 'HJ', 'CO', 'BTN', 'SB', 'BB'],
  };
  if (NAMED[nPlayers]) return NAMED[nPlayers];
  const all = ['UTG', 'UTG1', 'MP', 'LJ', 'HJ', 'CO', 'BTN'];
  return [...all.slice(7 - (nPlayers - 2)), 'SB', 'BB'];
}

const PRESETS = [
  {
    name: 'HU 10bb push/fold',
    players: 2, stack: 10, opens: '', mult: '', maxRaises: 1,
    limp: false, allin: true, ante: 0, rakePct: 0, rakeCap: 0,
  },
  {
    name: 'HU 25bb: limp, raise, jam',
    players: 2, stack: 25, opens: '2,2.5', mult: '3', maxRaises: 3,
    limp: true, allin: true, ante: 0, rakePct: 0, rakeCap: 0,
  },
  {
    name: '6-max 100bb, 2.5x open (no limps)',
    players: 6, stack: 100, opens: '2.5', mult: '3', maxRaises: 4,
    limp: false, allin: false, ante: 0, rakePct: 0, rakeCap: 0,
  },
  {
    name: '6-max 100bb low-stakes: limps + 5% rake',
    players: 6, stack: 100, opens: '2.5,4', mult: '3', maxRaises: 3,
    limp: true, allin: false, ante: 0, rakePct: 5, rakeCap: 3,
  },
  {
    // Matthew's live game. Deliberately bigger than the laptop limits —
    // the live estimate shows the real size; raise PREFLOP_MAX_NODES /
    // PREFLOP_MAX_ARENA_MB on a big machine, or trim the raise cap.
    name: '$2/2 8-max casino',
    players: 8, stack: 150, opens: '5,7.5,10', mult: '2,3.5,6', maxRaises: 3,
    limp: true, allin: false, ante: 0, rakePct: 10, rakeCap: 11,
  },
];

export function initPreflopLab({ els, onExport, toast }) {
  const S = {
    built: false,
    cursor: [],     // action indices to the node being VIEWED
    lineP: [],      // the full line (cursor is always a prefix of it)
    lineHist: null, // /api/preflop/node history for lineP (ribbon source)
    view: null,
    polling: null,
    positions: [],
    cells: [],      // persistent 13x13 cell divs (same markup as Browse)
    colors: [],
    fillMode: 'normalized', // 'normalized' | 'range' | 'full' (as in Browse)
    selected: null, // pinned [i, j]
    hover: null,
  };

  // Browse-identical matrix: same .cell markup/classes, hover shows the
  // detail strip, click pins a cell.
  (function buildGrid() {
    const m = els.grid;
    m.innerHTML = '';
    for (let i = 0; i < 13; i++) {
      for (let j = 0; j < 13; j++) {
        const cell = document.createElement('div');
        cell.className = 'cell';
        cell.innerHTML =
          `<div class="bars"></div><div class="fill"></div>` +
          `<div class="tag">${cellInfo(i, j).label}</div><div class="sub"></div>`;
        cell.addEventListener('click', () => {
          S.selected = S.selected && S.selected[0] === i && S.selected[1] === j
            ? null : [i, j];
          paintGrid();
          renderDetail();
        });
        cell.addEventListener('mouseenter', () => { S.hover = [i, j]; renderDetail(); });
        m.appendChild(cell);
        S.cells.push(cell);
      }
    }
    m.addEventListener('mouseleave', () => { S.hover = null; renderDetail(); });
  })();
  els.fillSeg.querySelectorAll('button').forEach(b =>
    b.addEventListener('click', () => {
      S.fillMode = b.dataset.f;
      els.fillSeg.querySelectorAll('button').forEach(x =>
        x.classList.toggle('active', x === b));
      paintGrid();
    }));

  // ----- config -----
  PRESETS.forEach((p, i) => {
    const o = document.createElement('option');
    o.value = i;
    o.textContent = p.name;
    els.preset.appendChild(o);
  });
  const applyPreset = (p) => {
    els.players.value = p.players;
    els.stack.value = p.stack;
    els.opens.value = p.opens;
    els.mult.value = p.mult;
    els.maxRaises.value = p.maxRaises;
    els.limp.checked = p.limp;
    els.allin.checked = p.allin;
    els.ante.value = p.ante;
    els.rakePct.value = p.rakePct;
    els.rakeCap.value = p.rakeCap;
  };
  els.preset.addEventListener('change', () => {
    applyPreset(PRESETS[+els.preset.value]);
    updateEstimate();
  });
  applyPreset(PRESETS[0]);

  // ----- live tree-size estimate -----
  async function updateEstimate() {
    let e;
    try { e = await api.pfEstimate(config()); }
    catch (err) { els.estimate.textContent = `\u26a0 ${err.message}`; els.estimate.classList.add('bad'); return; }
    const nodes = (+e.nodes).toLocaleString();
    const mb = e.arena_mb < 1 ? e.arena_mb.toFixed(1) : e.arena_mb.toFixed(0);
    const fmtBig = x => x >= 1e6 ? (x / 1e6).toFixed(1) + 'M'
      : x >= 1e4 ? Math.round(x / 1e3) + 'k' : (+x).toLocaleString();
    const caps = `this machine allows ${fmtBig(e.limit_nodes)} nodes / ` +
      (e.limit_arena_mb >= 1000 ? (e.limit_arena_mb / 1000).toFixed(1) + ' GB' : e.limit_arena_mb.toFixed(0) + ' MB');
    if (e.ok) {
      els.estimate.innerHTML =
        `tree \u2248 <b>${nodes}</b> nodes \u00b7 ${(+e.action_nodes).toLocaleString()} decisions \u00b7 ${mb} MB \u2014 fits \u2713 <span class="dim">(${caps})</span>`;
      els.estimate.classList.remove('bad');
    } else {
      els.estimate.innerHTML =
        `tree ${e.truncated ? '&gt;' : '\u2248'} <b>${nodes}</b> nodes \u00b7 ${mb} MB \u2014 too big: ${caps}. Trim sizes, raises, or limps \u2717`;
      els.estimate.classList.add('bad');
    }
  }
  let estT = null;
  const estSoon = () => { clearTimeout(estT); estT = setTimeout(updateEstimate, 350); };
  [els.players, els.stack, els.opens, els.mult, els.maxRaises,
   els.ante, els.limp, els.allin].forEach(el => {
    el.addEventListener('input', estSoon);
    el.addEventListener('change', estSoon);
  });
  updateEstimate();

  function config() {
    const n = +els.players.value;
    const positions = positionsFor(n);
    const posts = positions.map(p => (p === 'SB' ? 0.5 : p === 'BB' ? 1.0 : 0.0));
    const nums = s => s.split(',').map(x => parseFloat(x)).filter(x => x > 0);
    return {
      positions,
      stack: +els.stack.value || 100,
      posts,
      ante: +els.ante.value || 0,
      limp: els.limp.checked,
      open_raises: nums(els.opens.value),
      raise_mults: nums(els.mult.value),
      max_raises: +els.maxRaises.value || 1,
      add_allin: els.allin.checked,
      rake_pct: +els.rakePct.value || 0,
      rake_cap: +els.rakeCap.value || 0,
      no_flop_no_drop: true,
      realization: 'static',
    };
  }

  // ----- build / solve -----
  els.build.addEventListener('click', async () => {
    els.build.disabled = true;
    els.buildInfo.textContent = 'building… (first build computes the equity table, ~1 min)';
    try {
      const cfg = config();
      const info = await api.pfBuild(cfg);
      S.built = true;
      S.positions = cfg.positions;
      S.cursor = [];
      S.lineP = [];
      els.buildInfo.textContent =
        `${info.nodes.toLocaleString()} nodes · ${info.action_nodes.toLocaleString()} decision points · ${info.arena_mb.toFixed(0)} MB`;
      toast('preflop game built — SOLVE it');
      refresh();
      startPolling();
    } catch (e) { toast(e.message, true); els.buildInfo.textContent = ''; }
    els.build.disabled = false;
  });

  els.solve.addEventListener('click', async () => {
    if (!S.built) return toast('build a game first', true);
    try {
      await api.pfSolve({ iterations: 3000, check_every: 50, target_gap: 0.005 });
      startPolling();
    } catch (e) { toast(e.message, true); }
  });
  els.stop.addEventListener('click', () => api.pfStop().catch(() => {}));

  function startPolling() {
    if (S.polling) clearInterval(S.polling);
    S.polling = setInterval(poll, 1000);
    poll();
  }
  let lastIter = -1;
  async function poll() {
    let st;
    try { st = await api.pfStatus(); } catch { return; }
    if (!st.state) return;
    const gaps = st.gaps && st.gaps.length
      ? ` · BR gap ${st.gap_total.toFixed(4)} bb (${st.gaps.map(g => g.toFixed(3)).join(' / ')})`
      : '';
    els.status.textContent = `${st.state} · iter ${st.iteration}${gaps}`;
    els.solve.classList.toggle('hidden', st.state === 'running');
    els.stop.classList.toggle('hidden', st.state !== 'running');
    if (st.iteration !== lastIter && S.built) {
      lastIter = st.iteration;
      refresh(); // strategies moved: repaint current node
    }
    if (st.state !== 'running' && S.polling && st.iteration === lastIter) {
      clearInterval(S.polling);
      S.polling = null;
    }
  }

  // ----- node navigation / rendering -----
  async function refresh() {
    if (!S.built) return;
    try {
      const needLine = S.cursor.length !== S.lineP.length;
      const [view, lineView] = await Promise.all([
        api.pfNode(S.cursor),
        needLine ? api.pfNode(S.lineP) : Promise.resolve(null),
      ]);
      S.view = view;
      S.lineHist = (lineView || view).history;
    } catch (e) { toast(e.message, true); return; }
    renderRibbon();
    renderNode();
  }

  /** Chosen steps of the line up to the cursor: [{pos, label, kind}]. */
  function takenSteps(upTo) {
    const out = [];
    (S.lineHist || []).slice(0, upTo).forEach(h => {
      if (h.chosen != null && h.actions[h.chosen]) {
        const a = h.actions[h.chosen];
        out.push({ pos: h.actor_pos, label: a.label, kind: a.kind });
      }
    });
    return out;
  }

  // Browse-style ribbon: one segment per decision along the FULL line, every
  // available action as a chip with its frequency. Clicking the taken chip
  // (or the segment) views that point without losing the line; clicking a
  // different chip branches the line there.
  function renderRibbon() {
    const el = els.ribbon;
    el.innerHTML = '';
    const hist = S.lineHist || [];
    const cursor = S.cursor.length;
    hist.forEach((h, d) => {
      const seg = document.createElement('div');
      seg.className = 'hist-seg' + (d === cursor ? ' current' : '');
      const head = document.createElement('div');
      head.className = 'hist-head';
      if (h.kind === 'action') {
        head.innerHTML = `<span>${d === 0 ? 'PREFLOP · ' : ''}${h.actor_pos}</span><b>${h.pot.toFixed(1)}</b>`;
      } else {
        head.innerHTML = h.kind === 'pot_share'
          ? `<span>FLOP</span><b>${h.pot.toFixed(1)}</b>`
          : `<span>END</span><b>${h.pot.toFixed(1)}</b>`;
      }
      seg.appendChild(head);
      seg.dataset.tip = d === cursor
        ? 'The point you are viewing.'
        : 'Click to view this point (the line is kept).';
      seg.addEventListener('click', () => {
        if (d !== cursor) { S.cursor = S.lineP.slice(0, d); refresh(); }
      });
      if (h.kind === 'action') {
        h.actions.forEach((a, k) => {
          const chip = document.createElement('div');
          chip.className = 'hist-chip' + (h.chosen === k ? ' taken' : '');
          chip.textContent = `${a.label} · ${(a.freq * 100).toFixed(0)}%`;
          chip.dataset.tip = h.chosen === k
            ? `${h.actor_pos} takes ${a.label} ${(a.freq * 100).toFixed(1)}% of the time here — the line follows this action. Click to view the moment just after it.`
            : `${h.actor_pos}: ${a.label} ${(a.freq * 100).toFixed(1)}% of the time. Click to ${h.chosen == null ? 'take' : 'branch the line onto'} this action.`;
          chip.addEventListener('click', (e) => {
            e.stopPropagation();
            if (h.chosen === k) {
              S.cursor = S.lineP.slice(0, d + 1); // view after the taken action
            } else {
              S.lineP = [...S.lineP.slice(0, d), k]; // branch (or advance) here
              S.cursor = S.lineP.slice();
            }
            refresh();
          });
          seg.appendChild(chip);
        });
      } else {
        const chip = document.createElement('div');
        chip.className = 'hist-chip taken';
        chip.textContent = h.kind === 'pot_share' ? 'flop reached' : 'hand over';
        seg.appendChild(chip);
      }
      el.appendChild(seg);
    });
  }

  function actionColors(actions) {
    let r = 0;
    return actions.map(a => {
      if (COLORS[a.kind]) return COLORS[a.kind];
      const c = RAISE_SHADES[Math.min(r, RAISE_SHADES.length - 1)];
      r += 1;
      return a.kind === 'jam' ? RAISE_SHADES[3] : c;
    });
  }

  function renderNode() {
    const v = S.view;
    els.exportBtn.classList.add('hidden');

    // seats strip: who's live, what they've put in
    els.seats.innerHTML = v.positions.map((p, i) => {
      const dead = !v.live[i];
      const cur = v.kind === 'action' && v.actor === i;
      return `<span class="pfl-seat${dead ? ' dead' : ''}${cur ? ' cur' : ''}">${p} <small>${v.invested[i].toFixed(1)}</small></span>`;
    }).join('');
    els.pot.textContent = `pot ${v.pot.toFixed(1)} bb${v.spr != null ? ` · SPR ${v.spr.toFixed(1)}` : ''}`;

    if (v.kind === 'action') {
      const colors = actionColors(v.actions);
      // headline: who acts, and what (if anything) they're facing
      const past = takenSteps(S.cursor.length);
      const lastAggr = [...past].reverse().find(s => s.kind === 'raise' || s.kind === 'jam');
      const facing = lastAggr && lastAggr.pos !== v.actor_pos
        ? ` — facing ${lastAggr.pos}'s ${lastAggr.label}`
        : lastAggr && lastAggr.pos === v.actor_pos
          ? '' // their own raise came back around (someone called/limped behind)
          : past.length ? ' — unraised pot' : ' — first to act';
      els.nodeTitle.textContent = `${v.actor_pos} to act${facing} · pick actions in the ribbon above`;
      S.colors = colors;
      els.grid.classList.remove('hidden');
      els.fillSeg.classList.remove('hidden');
      paintGrid();
      renderDetail();
      renderLegend(v, colors);
      els.gridCap.innerHTML =
        `Grid = <b>${v.actor_pos}</b>'s play with every starting hand AT THIS POINT. ` +
        `Bar colors = how often the hand takes each action; <b>dim cells</b> = hands ` +
        `${v.actor_pos} rarely still holds here, filtered out by its own earlier actions ` +
        `(hover a cell for exact numbers).`;
    } else if (v.kind === 'fold_win') {
      const w = v.positions[v.live.findIndex(x => x)];
      els.nodeTitle.textContent = `everyone folded — ${w} takes ${v.pot.toFixed(1)} bb`;
      hideGrid();
    } else {
      const live = v.positions.filter((_, i) => v.live[i]);
      els.nodeTitle.textContent =
        `FLOP: ${live.join(' vs ')} · pot ${v.pot.toFixed(1)} bb` +
        (v.exportable ? '' : ` (${live.length}-way — postflop solver is heads-up only)`);
      hideGrid();
      if (v.exportable) {
        els.exportBtn.classList.remove('hidden');
      }
    }
  }

  els.exportBtn.addEventListener('click', async () => {
    try {
      const ex = await api.pfExport(S.cursor);
      const steps = takenSteps(S.cursor.length);
      const lineText = steps.map(s => `${s.pos} ${s.label}`).join(' · ') || 'root';
      // ribbon segments for Browse: continuing actions only (folds are just
      // dead money in the pot, same convention as the study module)
      ex.segments = steps
        .filter(st => st.kind !== 'fold')
        .map(st => ({ pos: st.pos, label: st.label }));
      onExport(ex, lineText);
    } catch (e) { toast(e.message, true); }
  });

  function hideGrid() {
    els.grid.classList.add('hidden');
    els.fillSeg.classList.add('hidden');
    els.detail.textContent = '';
    els.legend.innerHTML = '';
    els.gridCap.innerHTML = '';
  }

  /** Repaint the persistent cells from the current view (Browse STRAT style:
   *  discrete action colors at full opacity, reach shown as bottom-anchored
   *  bar height, empty cells dark with a dim label). */
  function paintGrid() {
    const v = S.view;
    if (!v || v.kind !== 'action') return;
    const colors = S.colors;
    const na = v.actions.length;
    let maxReach = 1e-9;
    for (let h = 0; h < 169; h++) maxReach = Math.max(maxReach, v.reach[h]);
    const fillH = r => {
      if (r <= 1e-9) return 0;
      if (S.fillMode === 'full') return 1;
      if (S.fillMode === 'range') return Math.min(1, r);
      return Math.min(1, r / maxReach);
    };
    for (let i = 0; i < 13; i++) {
      for (let j = 0; j < 13; j++) {
        const cell = S.cells[i * 13 + j];
        const idx = (12 - i) * 13 + (12 - j);
        const reach = v.reach[idx];
        const bars = cell.querySelector('.bars');
        const segs = [];
        for (let a = 0; a < na; a++) {
          const f = v.strategy[a * 169 + idx];
          if (f > 0.001) {
            segs.push(`<div style="width:${(f * 100).toFixed(1)}%;background:${colors[a]}"></div>`);
          }
        }
        bars.innerHTML = segs.join('');
        bars.style.height = `${(fillH(reach) * 100).toFixed(1)}%`;
        bars.style.opacity = reach > 1e-9 ? 1 : 0;
        cell.classList.toggle('empty', reach < 0.002);
        cell.classList.toggle('selected',
          !!(S.selected && S.selected[0] === i && S.selected[1] === j));
        cell.querySelector('.sub').textContent = '';
      }
    }
  }

  /** Detail strip for the hovered (or pinned) cell: exact action mix + how
   *  much of the class still reaches this node. */
  function renderDetail() {
    const v = S.view;
    const t = S.hover || S.selected; // pinned cell persists when the mouse leaves
    if (!v || v.kind !== 'action' || !t) {
      els.detail.textContent = '';
      return;
    }
    const [i, j] = t;
    const idx = (12 - i) * 13 + (12 - j);
    const lab = cellInfo(i, j).label;
    const reach = v.reach[idx];
    if (reach < 0.002) {
      els.detail.textContent = `${lab} — ${v.actor_pos} almost never holds this here`;
      return;
    }
    const mix = v.actions
      .map((a, k) => `${a.label} ${(v.strategy[k * 169 + idx] * 100).toFixed(1)}%`)
      .join(' · ');
    els.detail.textContent =
      `${lab} — ${mix}` + (reach < 0.995 ? ` · ${(reach * 100).toFixed(0)}% of combos reach` : '');
  }

  function renderLegend(v, colors) {
    els.legend.innerHTML = v.actions.map((a, k) =>
      `<span class="key"><i style="background:${colors[k]}"></i>${a.label}</span>`).join('');
  }

  return { refresh };
}
