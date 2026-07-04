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
    7: ['UTG', 'MP', 'HJ', 'CO', 'BTN', 'SB', 'BB'],
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
    players: 8, stack: 150, opens: '5,7.5,10', mult: '2,3.5', maxRaises: 2,
    limp: true, allin: true, ante: 0, rakePct: 10, rakeCap: 11,
  },
];

export function initPreflopLab({ els, onExport, toast, gotoSetup }) {
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
    rangeSeat: 0,   // whose arriving range the grid shows at terminals
    lastState: null, // last solver state seen by poll (drives the progress bar)
    gap0: null,      // first measured BR gap of the current run (progress scale)
  };

  // Browse-identical matrix: same .cell markup/classes; each cell carries a
  // data-tip with its exact numbers (the app tooltip system is delegated, so
  // updating the attribute on repaint is all it takes).
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
        m.appendChild(cell);
        S.cells.push(cell);
      }
    }
  })();

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
  let estSeq = 0;
  async function updateEstimate() {
    const seq = ++estSeq;
    let e;
    try { e = await api.pfEstimate(config()); }
    catch (err) {
      if (seq !== estSeq) return; // superseded by a newer request
      els.estimate.textContent = `\u26a0 ${err.message}`;
      els.estimate.classList.add('bad');
      return;
    }
    if (seq !== estSeq) return; // stale response: a newer one is coming
    const nodes = (+e.nodes).toLocaleString();
    const mb = e.arena_mb < 1 ? e.arena_mb.toFixed(1) : e.arena_mb.toFixed(0);
    const fmtBig = x => x >= 1e6 ? (x / 1e6).toFixed(1) + 'M'
      : x >= 1e4 ? Math.round(x / 1e3) + 'k' : (+x).toLocaleString();
    const caps = `this machine allows ${fmtBig(e.limit_nodes)} nodes / ` +
      (e.limit_arena_mb >= 1000 ? (e.limit_arena_mb / 1000).toFixed(1) + ' GB' : e.limit_arena_mb.toFixed(0) + ' MB');
    const borderline = e.ok &&
      (e.nodes > 0.9 * e.limit_nodes || e.arena_mb > 0.9 * e.limit_arena_mb);
    els.estimate.classList.toggle('warn', borderline);
    const capsTip = `${caps}. The caps track FREE RAM, so they move as other apps use memory ` +
      `(PREFLOP_MAX_NODES / PREFLOP_MAX_ARENA_MB env vars override). ` +
      `Tree size multiplies: open sizes \u00d7 re-raises \u00d7 raise cap \u00d7 limps \u00d7 players.`;
    const dot = t => ` <span class="info-dot" data-tip="${t}">?</span>`;
    if (e.ok && borderline) {
      els.estimate.classList.remove('bad');
      els.estimate.innerHTML =
        `tree \u2248 <b>${nodes}</b> nodes \u00b7 ${mb} MB \u2014 fits, <b>barely</b> \u26a0` +
        dot(capsTip + ' With this little headroom the build may still refuse if memory tightens \u2014 close big apps or trim a size.');
    } else if (e.ok) {
      els.estimate.classList.remove('bad');
      els.estimate.innerHTML =
        `tree \u2248 <b>${nodes}</b> nodes \u00b7 ${mb} MB \u2014 fits \u2713` + dot(capsTip);
    } else {
      els.estimate.classList.add('bad');
      els.estimate.innerHTML = (e.truncated
        ? `tree &gt; <b>${nodes}</b> nodes \u00b7 &gt; ${mb} MB \u2014 too big \u2717`
        : `tree \u2248 <b>${nodes}</b> nodes \u00b7 ${mb} MB \u2014 too big \u2717`) +
        dot((e.truncated ? 'Counting stopped early \u2014 hopelessly past the cap. ' : '') +
          capsTip + ' Trim open sizes, re-raise multipliers, the raise cap, or limps.');
    }
  }
  window.addEventListener('focus', () => estSoon());
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

  // One progress bar serves both phases. Build progress is an estimate:
  // expected node count from the size estimator divided by a build rate
  // that self-calibrates from every real build (localStorage). Solve
  // progress is real: iterations vs the requested maximum.
  const SOLVE_ITERS = 3000;
  const progFill = els.prog.querySelector('i');
  const progLab = els.prog.querySelector('span');
  function progressSet(pct, label) {
    els.prog.classList.remove('hidden');
    progFill.style.width = `${Math.max(0, Math.min(100, pct)).toFixed(1)}%`;
    progLab.textContent = label;
  }
  function progressHide() {
    els.prog.classList.add('hidden');
  }

  async function buildGame() {
    const cfg = config();
    els.build.disabled = true;
    els.solve.disabled = true;
    els.buildInfo.textContent = '';
    let expected = 0;
    try { expected = (await api.pfEstimate(cfg)).nodes || 0; } catch { /* build reports errors */ }
    const eqCold = !localStorage.getItem('pfl-eq-built');
    const rate = +localStorage.getItem('pfl-build-rate') || 150000; // nodes/s
    const t0 = performance.now();
    const tick = () => {
      const secs = (performance.now() - t0) / 1000;
      const pct = expected > 0 ? Math.min(94, (100 * secs * rate) / expected) : Math.min(94, secs * 12);
      progressSet(pct, eqCold
        ? 'building — the first run also computes the equity table (~15 s)…'
        : `building ${expected.toLocaleString()} nodes · ~${Math.round(pct)}%`);
    };
    tick();
    const timer = setInterval(tick, 150);
    try {
      const info = await api.pfBuild(cfg);
      const secs = (performance.now() - t0) / 1000;
      if (info.nodes > 20000 && secs > 0.2) {
        localStorage.setItem('pfl-build-rate', String(Math.round(info.nodes / secs)));
      }
      localStorage.setItem('pfl-eq-built', '1');
      S.built = true;
      S.builtCfg = JSON.stringify(cfg);
      S.positions = cfg.positions;
      S.cursor = [];
      S.lineP = [];
      els.buildInfo.textContent =
        `${info.nodes.toLocaleString()} nodes · ${info.action_nodes.toLocaleString()} decision points · ${info.arena_mb.toFixed(0)} MB`;
      progressSet(100, 'built ✓ — SOLVE to fill in the strategies');
      setTimeout(() => { if (S.lastState !== 'running') progressHide(); }, 1500);
      lastIter = 0;
      clearRightPanel(); // uniform pre-solve strategies aren't worth showing
      startPolling();
      return true;
    } catch (e) {
      toast(e.message, true);
      els.buildInfo.textContent = '';
      progressHide();
      updateEstimate(); // re-sync the size line with the caps that refused us
      return false;
    } finally {
      clearInterval(timer);
      els.build.disabled = false;
      els.solve.disabled = false;
    }
  }
  els.build.addEventListener('click', buildGame);

  // SOLVE builds first when there's nothing built yet or the settings
  // changed since the last build — one button does the right thing.
  els.solve.addEventListener('click', async () => {
    els.solve.disabled = true;
    try {
      if (!S.built || S.builtCfg !== JSON.stringify(config())) {
        if (!(await buildGame())) return;
      }
      progressSet(0, 'solving…');
      await api.pfSolve({ iterations: SOLVE_ITERS, check_every: 50, target_gap: 0.005 });
      startPolling();
    } catch (e) { toast(e.message, true); progressHide(); }
    finally { els.solve.disabled = false; }
  });
  els.stop.addEventListener('click', () => {
    els.stop.disabled = true;
    progLab.textContent = 'stopping…';
    api.pfStop().catch(() => {});
  });

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
    let gaps = '';
    if (st.gaps && st.gaps.length) {
      gaps = st.gaps.length <= 2
        ? ` · BR gap ${st.gap_total.toFixed(4)} bb (${st.gaps.map(g => g.toFixed(3)).join(' / ')})`
        : ` · BR gap ${st.gap_total.toFixed(4)} bb (worst seat ${Math.max(...st.gaps).toFixed(3)})`;
      els.status.dataset.tip =
        'Best-response gap: how much each seat could gain by deviating (bb) — the convergence metric. ' +
        st.gaps.map((g, i) => `${S.positions[i] || i}: ${g.toFixed(4)}`).join(' · ');
    }
    els.status.textContent = `${st.state} · iter ${st.iteration}${gaps}`;
    els.solve.textContent = st.state === 'done' || st.state === 'stopped' ? '3 · RE-SOLVE' : '3 · SOLVE';
    els.solve.classList.toggle('hidden', st.state === 'running');
    els.stop.classList.toggle('hidden', st.state !== 'running');
    if (st.state === 'running') {
      if (S.lastState !== 'running') S.gap0 = null;
      if (S.gap0 == null && st.gap_total > 0) S.gap0 = st.gap_total;
      // progress = the better of iteration count and log-scale gap convergence
      // (gap 0.005 bb is the finish line; iterations are the safety cap)
      const gapProg = S.gap0 > 0.005 && st.gap_total > 0
        ? Math.log(S.gap0 / st.gap_total) / Math.log(S.gap0 / 0.005) : 0;
      const pct = 100 * Math.max(st.iteration / SOLVE_ITERS, Math.min(1, Math.max(0, gapProg)));
      progressSet(pct,
        `solving · iter ${st.iteration}/${SOLVE_ITERS} · gap ${st.gap_total > 0 ? st.gap_total.toFixed(4) : '…'} → target 0.0050 bb`);
    } else if (S.lastState === 'running') {
      // a run just ended (target hit, max iterations, or STOP)
      els.stop.disabled = false;
      progressSet(100, st.state === 'done' ? 'solved ✓ (target gap reached or max iterations)' : 'stopped');
      setTimeout(() => { if (S.lastState !== 'running') progressHide(); }, 1200);
    }
    S.lastState = st.state;
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
  function clearRightPanel() {
    els.ribbon.innerHTML = '';
    els.nodeTitle.textContent = '';
    els.seats.innerHTML = '';
    els.exportBtn.classList.add('hidden');
    hideGrid();
  }

  async function refresh() {
    if (!S.built) return;
    if (lastIter < 1) { clearRightPanel(); return; } // nothing meaningful before solving
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
        head.innerHTML = `<span>${h.actor_pos}</span><b>${h.pot.toFixed(1)}</b>`;
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

    // seats strip: only while there's action (the ribbon carries the rest)
    els.seats.innerHTML = v.kind !== 'action' ? '' : v.positions.map((p, i) => {
      const dead = !v.live[i];
      const cur = v.actor === i;
      return `<span class="pfl-seat${dead ? ' dead' : ''}${cur ? ' cur' : ''}">${p} <small>${v.invested[i].toFixed(1)}</small></span>`;
    }).join('');

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
      els.rangeSeg.innerHTML = '';
      els.grid.classList.remove('hidden');
      els.fillSeg.classList.remove('hidden');
      paintGrid();
      renderLegend(v, colors);
      els.gridCap.innerHTML =
        `Grid = <b>${v.actor_pos}</b>'s play with every starting hand AT THIS POINT. ` +
        `Bar colors = how often the hand takes each action; <b>dim cells</b> = hands ` +
        `${v.actor_pos} rarely still holds here, filtered out by its own earlier actions ` +
        `(hover a cell for exact numbers).`;
    } else if (v.kind === 'fold_win') {
      const wi = v.live.findIndex(x => x);
      els.nodeTitle.textContent = `everyone folded — ${v.positions[wi]} takes ${v.pot.toFixed(1)} bb`;
      // still worth seeing: the range the winner got through with
      S.rangeSeat = wi;
      els.rangeSeg.innerHTML = '';
      els.grid.classList.remove('hidden');
      els.fillSeg.classList.remove('hidden');
      paintGrid();
      els.legend.innerHTML =
        `<span class="key"><i style="background:#f28c26"></i>${v.positions[wi]}'s range when everyone folds — bar height = share of combos</span>`;
      els.gridCap.innerHTML = '';
    } else {
      const live = v.positions.filter((_, i) => v.live[i]);
      els.nodeTitle.textContent =
        `FLOP: ${live.join(' vs ')} · pot ${v.pot.toFixed(1)} bb` +
        (v.spr != null ? ` · SPR ${v.spr.toFixed(1)}` : '');
      // keep the hand grid up: it shows each live player's arriving range
      const liveSeats = v.positions.map((_, i) => i).filter(i => v.live[i]);
      if (!liveSeats.includes(S.rangeSeat)) S.rangeSeat = liveSeats[0];
      els.rangeSeg.innerHTML = '';
      liveSeats.forEach(i => {
        const b = document.createElement('button');
        b.className = S.rangeSeat === i ? 'active' : '';
        b.textContent = v.positions[i];
        b.addEventListener('click', () => {
          S.rangeSeat = i;
          renderNode();
        });
        els.rangeSeg.appendChild(b);
      });
      els.grid.classList.remove('hidden');
      els.fillSeg.classList.remove('hidden');
      paintGrid();
      els.legend.innerHTML =
        `<span class="key"><i style="background:#f28c26"></i>${v.positions[S.rangeSeat]}'s arriving range — bar height = share of that hand's combos reaching this flop</span>`;
      els.gridCap.innerHTML = '';
      if (v.exportable) {
        els.exportBtn.classList.remove('hidden');
        els.gridCap.innerHTML =
          'The grid shows each player\u2019s arriving range \u2014 exactly the conditional ' +
          'ranges SEND TO POSTFLOP drops into SETUP, along with this pot and stack.';
      } else {
        // 3+ players see the flop: the postflop solver is heads-up only
        els.gridCap.innerHTML =
          `This line goes <b>${live.length}-way</b> to the flop, and the postflop solver ` +
          `is heads-up only. Branch the ribbon above onto a line where exactly two ` +
          `players see the flop (SEND TO POSTFLOP lights up there), or set a spot up ` +
          `manually with your own ranges in SETUP. `;
        const b = document.createElement('button');
        b.className = 'btn ghost';
        b.textContent = 'GO TO SETUP →';
        b.style.marginLeft = '6px';
        b.addEventListener('click', gotoSetup);
        els.gridCap.appendChild(b);
      }
    }
  }

  els.exportBtn.addEventListener('click', async () => {
    if (lastIter < 1) {
      return toast('solve the game first — until then every range is uniform', true);
    }
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
    els.rangeSeg.innerHTML = '';
    els.legend.innerHTML = '';
    els.gridCap.innerHTML = '';
  }

  /** Repaint the persistent cells from the current view (Browse STRAT style:
   *  discrete action colors at full opacity, reach shown as bottom-anchored
   *  bar height, empty cells dark with a dim label). */
  function paintGrid() {
    const v = S.view;
    if (!v) return;
    // action nodes paint the actor's strategy; flop terminals paint the
    // selected live player's ARRIVING RANGE (single-color, reach heights)
    let reachVec = null, na = 0;
    if (v.kind === 'action') {
      reachVec = v.reach;
      na = v.actions.length;
    } else if (v.kind === 'pot_share' || v.kind === 'fold_win') {
      reachVec = (v.reaches_all || [])[S.rangeSeat] || null;
    }
    if (!reachVec) return;
    const colors = S.colors;
    // bar height = the actual fraction of the hand's combos still held here
    const fillH = r => (r <= 1e-9 ? 0 : Math.min(1, r));
    for (let i = 0; i < 13; i++) {
      for (let j = 0; j < 13; j++) {
        const cell = S.cells[i * 13 + j];
        const idx = (12 - i) * 13 + (12 - j);
        const reach = reachVec[idx] || 0;
        const bars = cell.querySelector('.bars');
        const segs = [];
        if (v.kind === 'action') {
          for (let a = 0; a < na; a++) {
            const f = v.strategy[a * 169 + idx];
            if (f > 0.001) {
              segs.push(`<div style="width:${(f * 100).toFixed(1)}%;background:${colors[a]}"></div>`);
            }
          }
        } else {
          segs.push('<div style="width:100%;background:#f28c26"></div>');
        }
        bars.innerHTML = segs.join('');
        bars.style.height = `${(fillH(reach) * 100).toFixed(1)}%`;
        bars.style.opacity = reach > 1e-9 ? 1 : 0;
        cell.classList.toggle('empty', reach < 0.002);
        const lab = cellInfo(i, j).label;
        if (v.kind === 'action') {
          cell.dataset.tip = reach < 0.002
            ? `${lab} — ${v.actor_pos} almost never holds this here`
            : `${lab} — ` + v.actions.map((a, k) =>
                `${a.label} ${(v.strategy[k * 169 + idx] * 100).toFixed(1)}%`).join(' · ') +
              (reach < 0.995 ? ` · ${(reach * 100).toFixed(0)}% of combos still in range` : '');
        } else {
          const pos = v.positions[S.rangeSeat];
          cell.dataset.tip = reach < 0.002
            ? `${lab} — not in ${pos}'s range here`
            : `${lab} — ${(reach * 100).toFixed(0)}% of ${pos}'s combos still held here`;
        }
        cell.querySelector('.sub').textContent = '';
      }
    }
  }


  function renderLegend(v, colors) {
    els.legend.innerHTML =
      `<span class="key dim">cell colors = ${v.actor_pos}'s action mix:</span>` +
      v.actions.map((a, k) =>
        `<span class="key"><i style="background:${colors[k]}"></i>${a.label}</span>`).join('') +
      `<span class="key dim">\u00b7 bar height = share of the hand's combos still in range \u00b7 dark cell = hand no longer here</span>`;
  }

  return { refresh };
}
