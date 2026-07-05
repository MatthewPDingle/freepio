// Strategy browser: navigate the tree, render the 13x13 strategy matrix,
// action frequencies, per-combo detail, EV and equity displays.

import { cellInfo, cellCombos, comboIndex, cardToString, cardFromString,
         rank, suit, RANKS, SUIT_GLYPH, SUITS } from './cards.js';
import { api, toast } from './api.js';
import { classify, suitTags, MADE_LABELS, MADE_ORDER, DRAW_LABELS, DRAW_ORDER,
         EQS_LABELS, EQA_LABELS } from './classify.js';

// Strategy color palette.
const ACTION_COLORS = {
  fold: '#4a78c8',
  check: '#5ca75f',
  call: '#5ca75f',
};
// Bet/raise reds by size category (Small / Medium / Large / Overbet),
// classified by the wager as a fraction of the pot.
const BET_SHADES = {
  small: '#e8484c',
  medium: '#c24345',
  large: '#a23a3c',
  overbet: '#7c3134',
};
// Player colors: OOP light cyan, IP green.
const EQ_COLORS = ['#8edced', '#46b556'];
// Equity chart margins (CSS px).
const EQ_M = { L: 38, R: 10, T: 12, B: 24 };

function betShade(amount, pot) {
  const pct = pot > 0 ? (amount / pot) * 100 : 100;
  if (pct <= 40) return BET_SHADES.small;
  if (pct <= 80) return BET_SHADES.medium;
  if (pct <= 135) return BET_SHADES.large;
  return BET_SHADES.overbet;
}

function stepEq(a, b) {
  if (!a || !b || a.type !== b.type) return false;
  return a.type === 'action' ? a.index === b.index : a.card === b.card;
}
function isPrefix(a, b) {
  return a.length <= b.length && a.every((s, i) => stepEq(s, b[i]));
}

export class Browser {
  constructor(els) {
    this.els = els;
    this.path = [];        // the node currently VIEWED (a prefix of `line`)
    this.line = [];        // the full navigated line (cursor can sit anywhere on it)
    this.lineHistory = null; // cached node history for the whole line (ribbon source)
    this.preflop = null;   // {oop, ip, potBb, effStackBb, segments} from preflop study setup, else null
    this.view = null;
    this.player = 0;     // matrix viewpoint
    this.mode = 'strategy';
    this.exploit = null;          // /api/exploit payload for EXPLOIT mode
    this._exploitHands = null;    // view-shaped cache of exploit.hands
    this._exploitLoading = false;
    this.comboRows = false; // orientation: false = Horizontal (combined bar), true = Vertical (per-combo columns)
    this.fillMode = 'normalized'; // 'normalized' | 'range' | 'full'
    this.selectedCell = null; // pinned by click
    this.hoverCell = null;    // transient mouse-over
    this.handsTab = 'hands';
    // Filters tab state (persists across navigation)
    this.filterMode = 'include';
    this.filterCats = new Set();  // made/draw/eq bucket keys
    this.filterSuits = new Set(); // s0..s3 (suited), o0..o3 (offsuit containing)
    this.filterPreview = null;    // hovered filter: {type:'cat'|'suit', key}
    this.actionFilter = null;     // selected action tile -> grid shows its range
    this.buildMatrix();
    if (this.els.eqCanvas) {
      this.els.eqCanvas.addEventListener('mousemove', e => {
        const r = this.els.eqCanvas.getBoundingClientRect();
        const x = (e.clientX - r.left - EQ_M.L) / (r.width - EQ_M.L - EQ_M.R);
        this.eqHoverX = Math.max(0, Math.min(1, x));
        this.drawEquityChart();
      });
      this.els.eqCanvas.addEventListener('mouseleave', () => {
        this.eqHoverX = null;
        this.drawEquityChart();
      });
    }
    if (this.els.handsTabs) {
      this.els.handsTabs.querySelectorAll('.htab').forEach(b =>
        b.addEventListener('click', () => {
          this.handsTab = b.dataset.v;
          this.els.handsTabs.querySelectorAll('.htab').forEach(x =>
            x.classList.toggle('active', x.dataset.v === this.handsTab));
          this.renderHandsPanel();
        }));
    }
    // action-history ribbon scrolling: arrows, mouse wheel, and arrow refresh
    if (this.els.histLeft) {
      const step = () => Math.max(140, this.els.history.clientWidth * 0.7);
      this.els.histLeft.addEventListener('click', () =>
        this.els.history.scrollBy({ left: -step(), behavior: 'smooth' }));
      this.els.histRight.addEventListener('click', () =>
        this.els.history.scrollBy({ left: step(), behavior: 'smooth' }));
      this.els.history.addEventListener('scroll', () => this.updateHistNav());
      this.els.history.addEventListener('wheel', e => {
        // let a vertical wheel scroll the ribbon horizontally
        if (Math.abs(e.deltaY) > Math.abs(e.deltaX)) {
          this.els.history.scrollLeft += e.deltaY;
          e.preventDefault();
        }
      }, { passive: false });
      window.addEventListener('resize', () => this.updateHistNav());
    }
    window.addEventListener('resize', () => this.drawEquityChart());
    // keyboard: Backspace = step back one node, Escape = unpin
    document.addEventListener('keydown', e => {
      const browseActive = document.getElementById('view-browse')?.classList.contains('active');
      const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName || '');
      if (!browseActive || typing || !this.view) return;
      if (e.key === 'Backspace' && this.path.length) {
        e.preventDefault();
        this.viewDepth(this.path.length - 1); // step cursor back, keep the line
      } else if (e.key === 'Escape' && this.selectedCell) {
        this.selectedCell = null;
        this.renderMatrix();
        this.renderHandsPanel();
        this.drawEquityChart();
      }
    });
  }

  buildMatrix() {
    const m = this.els.matrix;
    m.innerHTML = '';
    this.cells = [];
    for (let i = 0; i < 13; i++) {
      for (let j = 0; j < 13; j++) {
        const info = cellInfo(i, j);
        const cell = document.createElement('div');
        cell.className = 'cell';
        cell.dataset.i = i; cell.dataset.j = j;
        cell.innerHTML = `<div class="bars"></div><div class="fill"></div><div class="tag">${info.label}</div><div class="sub"></div>`;
        cell.addEventListener('click', () => this.selectCell(i, j));
        cell.addEventListener('mouseenter', () => {
          this.hoverCell = [i, j];
          this.renderHandsPanel(); // flips FILTERS/BLOCKERS to HANDS while hovering
          this.drawEquityChart();
          this.showHandPop(i, j, cell);
        });
        m.appendChild(cell);
        this.cells.push(cell);
      }
    }
    m.addEventListener('mouseleave', () => {
      this.hoverCell = null;
      this.renderHandsPanel(); // reverts to FILTERS/BLOCKERS when leaving the grid
      this.drawEquityChart();
      this.hideHandPop();
    });
  }

  async refresh() {
    // loading feedback: node EV queries can take a couple of seconds on big trees
    this.hideHandPop(); // stale combo popup would show the previous node
    if (this.els.pot) this.els.pot.textContent = 'computing…';
    if (this.els.matrix) this.els.matrix.style.opacity = '.55';
    try {
      this.view = await api.node(this.path);
      this.exploit = null;        // stale for the new node/strategy state
      this._exploitHands = null;
    } catch (e) {
      toast(`node error: ${e.message}`, true);
      if (this.path.length) { this.path = []; this.line = []; this.lineHistory = null; return this.refresh(); }
      this.view = null;
      return;
    }
    // Maintain the full line: viewing a node ON the line keeps the line intact;
    // a path that isn't a prefix of the line (a new/diverging action) replaces it.
    if (!this.lineHistory || !isPrefix(this.path, this.line)) {
      this.line = this.path.slice();
      this.lineHistory = this.view.history;
    }
    // index hands by combo for matrix lookup
    this.handIdx = [new Map(), new Map()];
    for (const p of [0, 1]) {
      this.view.players[p].hands.forEach((h, i) => {
        this.handIdx[p].set(comboIndex(h.c1, h.c2), i);
      });
    }
    // classify every hand for the Filters tab (cheap, done per node)
    const boardCards = this.view.board.map(cardFromString);
    this.boardSet = new Set(boardCards); // for board-aware combo-count normalization
    this.cats = [0, 1].map(p =>
      this.view.players[p].hands.map(h => ({
        ...classify(boardCards, h.c1, h.c2, h.eq),
        suits: suitTags(h.c1, h.c2),
      })));
    // auto-follow actor: matrix shows the player to act when at action node
    if (this.view.node_type === 'action') this.player = this.view.player;
    this.blockerSort = null; // action list may differ per node
    this.filterPreview = null;
    this.actionFilter = null; // action set differs per node
    // path labels (for lock descriptions) reconstructed from server history
    this.pathLabels = (this.view.history || [])
      .slice(0, this.path.length)
      .map(h => h.kind === 'card' ? (h.card || '?') : (h.actions[h.chosen]?.label || '?'));
    this.renderHistory();
    this.renderVillainLocks();
    this.renderActions();
    this.renderMatrix();
    this.renderLegend();
    this.renderHandsPanel();
    this.buildEqCurves();
    this.renderEqStats();
    this.drawEquityChart();
    this.applyFilterPreview(); // clears any stale preview dim
    this.syncSegs();
    this.els.matrix.style.opacity = '';
    this.els.pot.textContent =
      `pot ${fmt(this.view.pot)} · ${this.posLabel(0)} in ${fmt(this.view.put[0])} / ${this.posLabel(1)} in ${fmt(this.view.put[1])}`;
  }

  actionColors() {
    if (!this.view || this.view.node_type !== 'action') return [];
    return computeActionColors(this.view.actions, this.view.pot);
  }

  // ----- action history bar -----

  /** Reset the line + cursor (call when a new spot is built/loaded). */
  reset() {
    this.path = [];
    this.line = [];
    this.lineHistory = null;
    this.preflop = null; // build handler re-sets this after reset when applicable
    this.villainLocked = null; // player index whose postflop profile is locked in
  }

  /** Position label for player p: BB/BTN/… from the preflop study setup, else
   *  the generic OOP/IP. Used everywhere the two players are named. */
  posLabel(p) {
    if (this.preflop) return p === 0 ? this.preflop.oop : this.preflop.ip;
    return p === 0 ? 'OOP' : 'IP';
  }

  /** Take a step from node depth `d`: follow the line if it matches, else branch. */
  navigate(path) {
    this.path = path;
    this.refresh();
  }

  /** Move the view cursor to node `depth` along the line WITHOUT changing the line. */
  viewDepth(depth) {
    this.navigate(this.line.slice(0, depth));
  }

  renderHistory() {
    const el = this.els.history;
    if (!el) return;
    el.innerHTML = '';
    // Leading PREFLOP segments from the study setup (read-only — they describe
    // the line that produced this spot, not nodes in the solved postflop tree).
    if (this.preflop && this.preflop.segments) {
      for (const sg of this.preflop.segments) {
        const s = document.createElement('div');
        s.className = 'hist-seg hist-preflop';
        s.dataset.tip = 'Preflop action that produced this spot (from the preflop study panel or the PREFLOP LAB) — context only, not part of the solved postflop tree.';
        const h = document.createElement('div');
        h.className = 'hist-head';
        h.innerHTML = `<span>${sg.pos}</span>`;
        s.appendChild(h);
        const chip = document.createElement('div');
        chip.className = 'hist-chip taken';
        chip.textContent = sg.label;
        s.appendChild(chip);
        el.appendChild(s);
      }
    }
    // Render the FULL line (so past/future nodes stay visible); the cursor is
    // at depth = this.path.length. Viewing a node never truncates the line.
    const hist = this.lineHistory || this.view.history || [];
    const cursor = this.path.length;
    const STREETS = ['FLOP', 'TURN', 'RIVER'];

    const seg = (depth, cls = '') => {
      const d = document.createElement('div');
      d.className = `hist-seg ${cls}`;
      // clicking the segment body views that node (keeps the line intact)
      if (depth != null) {
        d.addEventListener('click', () => { if (depth !== cursor) this.viewDepth(depth); });
      }
      el.appendChild(d);
      return d;
    };
    const head = (parent, left, right) => {
      const h = document.createElement('div');
      h.className = 'hist-head';
      h.innerHTML = `<span>${left}</span>${right != null ? `<b>${right}</b>` : ''}`;
      parent.appendChild(h);
      return h;
    };

    // Leading board segment: the initial street's cards + starting pot.
    const cardStepsViewed = (this.view.history || []).filter(h => h.kind === 'card' && h.card).length;
    const initLen = this.view.board.length - cardStepsViewed;
    const rootPot = hist.length ? hist[0].pot : this.view.pot;
    {
      const s = seg(null);
      s.classList.add('hist-board');
      s.dataset.tip = 'The starting board. Click any past node to view it without losing the rest of the line.';
      head(s, STREETS[Math.max(0, initLen - 3)], fmt(rootPot));
      const row = document.createElement('div');
      row.className = 'hist-cards';
      for (let k = 0; k < initLen; k++) {
        row.appendChild(cardChip(this.view.board[k], 'bcard mini'));
      }
      s.appendChild(row);
      s.addEventListener('click', () => { if (cursor !== 0) this.viewDepth(0); });
    }

    hist.forEach((h, i) => {
      const prefix = this.line.slice(0, i);
      const isCursor = i === cursor;
      if (h.kind === 'action') {
        const s = seg(i, isCursor ? 'current' : '');
        s.dataset.tip = `${this.posLabel(h.player)} decision. Click to view this node; click a different action to branch the line here.`;
        head(s, this.posLabel(h.player), fmt(h.stack));
        h.actions.forEach((a, k) => {
          const chip = document.createElement('button');
          chip.className = 'hist-chip' + (h.chosen === k ? ' taken' : '');
          chip.textContent = a.label;
          chip.dataset.tip = h.chosen === k
            ? 'The action taken in this line — click to view this node.'
            : `Branch the line: take ${a.label.toLowerCase()} from here instead.`;
          chip.addEventListener('click', (e) => {
            e.stopPropagation();
            // taken action on a past node -> just view it (keep the line);
            // a different action (or any action at the frontier) -> branch/advance
            if (h.chosen === k) this.viewDepth(i);
            else this.navigate([...prefix, { type: 'action', index: k }]);
          });
          s.appendChild(chip);
        });
      } else if (h.kind === 'card') {
        const s = seg(i, isCursor ? 'current' : '');
        s.dataset.tip = 'Card dealt. Click to view this point; change the card from the panel below.';
        head(s, STREETS[h.street], fmt(h.pot));
        const row = document.createElement('div');
        row.className = 'hist-cards';
        row.appendChild(h.card ? cardChip(h.card, 'bcard mini') : facedownChip('bcard mini'));
        s.appendChild(row);
      } else {
        const s = seg(i, isCursor ? 'current dim' : 'dim');
        head(s, 'END', null);
        const lbl = document.createElement('div');
        lbl.className = 'hist-chip taken';
        // fold vs showdown: HistoryStep doesn't carry it, so look at the
        // action that led here
        const prev = hist[i - 1];
        const folded = prev && prev.kind === 'action'
          && prev.actions[prev.chosen]?.kind === 'fold';
        lbl.textContent = folded ? 'Fold — hand over' : 'Showdown';
        s.appendChild(lbl);
      }
    });

    // Trailing hint: next street still to come on the LINE's deepest node.
    const deep = hist[hist.length - 1];
    if (deep && deep.kind === 'action' && deep.street < 2) {
      const s = seg(null, 'dim');
      head(s, STREETS[deep.street + 1], null);
      const row = document.createElement('div');
      row.className = 'hist-cards';
      row.appendChild(facedownChip('bcard mini'));
      s.appendChild(row);
    }

    // keep the viewed node's segment in view, then refresh the scroll arrows
    const cur = el.querySelector('.hist-seg.current');
    if (cur) {
      const cr = cur.getBoundingClientRect(), er = el.getBoundingClientRect();
      if (cr.left < er.left) el.scrollLeft -= er.left - cr.left + 8;
      else if (cr.right > er.right) el.scrollLeft += cr.right - er.right + 8;
    }
    this.updateHistNav();
  }

  /** Show the ribbon scroll arrows only when (and on the side) it overflows. */
  updateHistNav() {
    const el = this.els.history;
    if (!el || !this.els.histLeft) return;
    const max = el.scrollWidth - el.clientWidth;
    const overflow = max > 2;
    this.els.histLeft.classList.toggle('hidden', !overflow || el.scrollLeft <= 1);
    this.els.histRight.classList.toggle('hidden', !overflow || el.scrollLeft >= max - 1);
  }

  // ----- actions panel -----

  renderActions() {
    const el = this.els.actionList;
    const picker = this.els.cardPicker;
    el.innerHTML = '';
    picker.classList.add('hidden');
    this.els.runouts && (this.els.runouts.innerHTML = '');
    this.runoutRep = null; // drop any prior runouts report when the node changes
    if (this.mode === 'exploit') this.renderExploitBanner(el);

    if (this.view.node_type === 'action') {
      const colors = this.actionColors();
      const actor = this.view.player;
      const hands = this.handsFor(actor);
      this.els.actionsTitle.textContent =
        `${this.posLabel(actor)} to act — street ${['flop','turn','river'][this.view.street]}`;
      // global frequencies: reach-weighted average strategy
      let totalReach = 0;
      const freqs = this.view.actions.map(() => 0);
      const evs = this.view.actions.map(() => ({ n: 0, d: 0 }));
      hands.forEach(h => {
        if (!h.strategy) return;
        totalReach += h.reach;
        h.strategy.forEach((s, a) => {
          freqs[a] += s * h.reach;
          if (h.evs && h.evs[a] != null) { evs[a].n += h.evs[a] * h.reach * s; evs[a].d += h.reach * s; }
        });
      });
      // Action tiles. Navigation lives in the history ribbon;
      // clicking a tile filters the grid to that action's range instead.
      const tiles = document.createElement('div');
      tiles.className = 'act-tiles';
      this.view.actions.forEach((a, k) => {
        const freq = totalReach > 0 ? freqs[k] / totalReach : 0;
        const ev = evs[k].d > 1e-9 ? evs[k].n / evs[k].d : null;
        const tile = document.createElement('div');
        tile.className = 'act-tile' + (this.actionFilter === k ? ' sel' : '');
        tile.style.background = colors[k];
        tile.dataset.tip = `${a.label}: ${(freq * 100).toFixed(1)}% of the range, ${freqs[k].toFixed(1)} combos${ev != null ? `, avg EV ${fmt(ev)}` : ''}. Click to show only this action's range on the grid.`;
        tile.innerHTML = `
          <div class="at-label">${a.label}</div>
          <div class="at-foot">
            <span class="at-pct">${(freq * 100).toFixed(1)}%</span>
            <span class="at-combos">${freqs[k].toFixed(1)}<small>combos</small></span>
          </div>`;
        tile.addEventListener('click', () => {
          this.actionFilter = this.actionFilter === k ? null : k;
          this.renderActions();
          this.renderMatrix();
          this.renderLegend();
        });
        tiles.appendChild(tile);
      });
      el.appendChild(tiles);
      // overall split bar
      const sumbar = document.createElement('div');
      sumbar.className = 'act-sumbar';
      sumbar.innerHTML = this.view.actions.map((_, k) => {
        const freq = totalReach > 0 ? freqs[k] / totalReach : 0;
        return `<i style="width:${(freq * 100).toFixed(2)}%;background:${colors[k]}"></i>`;
      }).join('');
      el.appendChild(sumbar);
      if (this.actionFilter != null) {
        const hint = document.createElement('div');
        hint.className = 'act-filter-hint';
        hint.textContent = `grid showing: ${this.view.actions[this.actionFilter].label} frequency · click the tile again to clear`;
        el.appendChild(hint);
      }
      this.renderLockControls(el, colors);
    } else if (this.view.node_type === 'chance') {
      this.els.actionsTitle.textContent =
        `dealing ${this.view.street === 1 ? 'turn' : 'river'} — pick a card`;
      picker.classList.remove('hidden');
      picker.innerHTML = '';
      const avail = new Set(this.view.available_cards);
      // one row per suit (s/h/d/c), one column per rank (A..2)
      for (const s of [3, 2, 1, 0]) {
        for (let r = 12; r >= 0; r--) {
          const cs = RANKS[r] + SUITS[s];
          const b = document.createElement('button');
          b.className = 'pick';
          const glyph = SUIT_GLYPH[SUITS[s]];
          b.innerHTML = `${RANKS[r]}<span class="suit-${SUITS[s]}">${glyph}</span>`;
          if (!avail.has(cs)) b.classList.add('used');
          b.addEventListener('click', () => {
            this.path.push({ type: 'card', card: cs });
            this.pathLabels.push(cs);
            this.refresh();
          });
          picker.appendChild(b);
        }
      }
      // runouts report
      const btn = document.createElement('button');
      btn.className = 'btn';
      btn.style.marginTop = '12px';
      btn.textContent = 'RUNOUTS REPORT';
      btn.dataset.tip = 'Strategy and equity for every possible next card at once — spot which cards favor which player and which runouts get barreled.';
      btn.addEventListener('click', () => this.loadRunouts(btn));
      el.appendChild(btn);
    } else {
      this.els.actionsTitle.textContent = 'Terminal';
      const banner = document.createElement('div');
      banner.className = 'terminal-banner';
      banner.textContent = this.view.node_type === 'terminal_fold'
        ? 'fold — hand over' : 'showdown';
      el.appendChild(banner);
    }
  }

  renderLockControls(el, colors) {
    const p = this.view.player;
    const acts = this.view.actions;
    const na = acts.length;
    const hands = this.view.players[p].hands;
    const locked = this.view.locked;
    const label = this.pathLabels?.length
      ? this.view.board.join('') + ' ' + this.pathLabels.join(' > ') : 'root';
    if (this.lockMode !== 'hands') this.lockMode = 'range';

    // current reach-weighted aggregate frequency per action
    let tot = 0;
    const agg = acts.map(() => 0);
    hands.forEach(h => {
      if (!h.strategy) return;
      tot += h.reach;
      h.strategy.forEach((s, k) => agg[k] += s * h.reach);
    });
    const curFreq = agg.map(x => tot > 1e-9 ? x / tot : 1 / na);

    const wrap = document.createElement('div');
    wrap.className = 'lock-panel';
    wrap.innerHTML = `
      <div class="lock-head">NODE LOCKING <span class="info-dot" data-tip="A lock pins THIS node's strategy so the solver stops adapting it. Two ways to use it: (1) RE-SOLVE — every other node re-optimizes around your assumption ('how should I play if villain really folds 70% here?'). (2) EXPLOIT mode — see the best response to the locked strategy immediately, no re-solve. Locks persist until UNLOCK, cover all suit-equivalent runouts, and re-locking replaces the old lock (never compounds).">?</span>${locked ? ' <span class="lock-badge">LOCKED</span>' : ''}</div>
      <div class="lock-modes seg">
        <button data-lm="range" class="${this.lockMode === 'range' ? 'active' : ''}"
          data-tip="Model a read about the whole range: 'folds 70% to this bet'. You set aggregate action frequencies and the solved strategy is rebalanced to hit them — each hand's mix is scaled proportionally, so the RIGHT hands do the extra folding/calling. This is the mode for pool tendencies.">Overall %</button>
        <button data-lm="hands" class="${this.lockMode === 'hands' ? 'active' : ''}"
          data-tip="Surgical edits: set exact action frequencies for one hand class (click a cell to pin it). Every other hand keeps its current strategy. Use when the read is hand-specific — 'always slowplays sets here' — rather than about the whole range.">Per hand</button>
      </div>
      <div class="lock-body"></div>`;
    const body = wrap.querySelector('.lock-body');
    wrap.querySelectorAll('.lock-modes button').forEach(b =>
      b.addEventListener('click', () => { this.lockMode = b.dataset.lm; this.renderActions(); }));

    // a per-action [ %] editor with a live total + normalize button
    const freqEditor = (init) => {
      const cont = document.createElement('div');
      cont.className = 'lk-actions';
      const inputs = [];
      acts.forEach((a, k) => {
        const row = document.createElement('label');
        row.className = 'lk-act';
        row.innerHTML = `<i class="lk-sw" style="background:${colors[k]}"></i><span class="lk-lab">${a.label}</span>`;
        const inp = document.createElement('input');
        inp.type = 'number'; inp.min = '0'; inp.max = '100'; inp.step = '1';
        inp.value = `${Math.round(init[k] * 100)}`;
        inp.className = 'lk-pct';
        const unit = document.createElement('span'); unit.className = 'lk-unit'; unit.textContent = '%';
        row.appendChild(inp); row.appendChild(unit);
        cont.appendChild(row);
        inputs.push(inp);
      });
      const sumRow = document.createElement('div');
      sumRow.className = 'lk-sumrow';
      const sumEl = document.createElement('span'); sumEl.className = 'lk-sum';
      const norm = document.createElement('button'); norm.className = 'btn ghost xs'; norm.textContent = 'normalize';
      const update = () => {
        const s = inputs.reduce((a, i) => a + (+i.value || 0), 0);
        sumEl.textContent = `total ${Math.round(s)}%`;
        sumEl.classList.toggle('bad', Math.abs(s - 100) > 0.5);
      };
      norm.addEventListener('click', () => {
        const s = inputs.reduce((a, i) => a + (+i.value || 0), 0);
        if (s > 0) inputs.forEach(i => { i.value = `${Math.round((+i.value || 0) / s * 100)}`; });
        update();
      });
      inputs.forEach(i => i.addEventListener('input', update));
      update();
      sumRow.appendChild(sumEl); sumRow.appendChild(norm);
      return { cont, inputs, sumRow };
    };

    if (this.lockMode === 'range') {
      const help = document.createElement('div');
      help.className = 'lk-help';
      help.innerHTML = 'Set how often the <b>whole range</b> takes each action, then LOCK FREQUENCIES. ' +
        'The solved strategy is rebalanced to hit your numbers (the pre-filled values are the current solution). ' +
        'Then <b>RE-SOLVE</b> to make the rest of the tree adapt to the read \u2014 or open <b>EXPLOIT</b> to punish it directly, no re-solve needed.';
      const ed = freqEditor(curFreq);
      body.append(help, ed.cont, ed.sumRow);
      const foot = document.createElement('div'); foot.className = 'btn-row';
      const lock = document.createElement('button'); lock.className = 'btn'; lock.textContent = 'LOCK FREQUENCIES';
      lock.dataset.tip = 'Pin this node to YOUR numbers (a read/assumption). Rebalances the solved strategy so the range hits these frequencies. Note: hands that never took an action in the solution can\u2019t be scaled into it \u2014 force those with Per hand.';
      lock.addEventListener('click', async () => {
        const freqs = ed.inputs.map(i => Math.max(0, +i.value || 0));
        if (freqs.reduce((a, b) => a + b, 0) <= 0) return toast('set at least one action above 0%', true);
        try {
          await api.lock(this.path, { kind: 'range', freqs }, label);
          toast('locked — RE-SOLVE to adapt the tree, or open EXPLOIT to attack it as-is'); this.refresh();
        } catch (e) { toast(e.message, true); }
      });
      foot.appendChild(lock);
      this.appendLockFooter(foot, label, locked);
      body.appendChild(foot);
    } else {
      const sel = this.selectedCell;
      const foot = document.createElement('div'); foot.className = 'btn-row';
      if (!sel) {
        const help = document.createElement('div'); help.className = 'lk-help';
        help.textContent = 'Click a hand in the matrix to pin it, then set its exact frequencies here.';
        body.appendChild(help);
      } else {
        const info = cellInfo(sel[0], sel[1]);
        let ct = 0; const cagg = acts.map(() => 0); const combos = [];
        for (const [a, b] of cellCombos(info)) {
          const hi = this.handIdx[p].get(comboIndex(a, b));
          if (hi === undefined) continue;
          combos.push(cardToString(a) + cardToString(b));
          const h = hands[hi];
          if (h.strategy) { ct += h.reach; h.strategy.forEach((s, k) => cagg[k] += s * h.reach); }
        }
        if (!combos.length) {
          body.innerHTML = `<div class="lk-help">${info.label} is not in ${this.posLabel(p)}'s range here.</div>`;
        } else {
          const cur = cagg.map(x => ct > 1e-9 ? x / ct : 1 / na);
          const help = document.createElement('div'); help.className = 'lk-help';
          help.innerHTML = `Set exact frequencies for <b style="color:var(--ink)">${info.label}</b> (${combos.length} combo${combos.length > 1 ? 's' : ''}). Other hands are unchanged.`;
          const ed = freqEditor(cur);
          body.append(help, ed.cont, ed.sumRow);
          const lock = document.createElement('button'); lock.className = 'btn'; lock.textContent = `LOCK ${info.label}`;
          lock.dataset.tip = 'Pin exact frequencies for just this hand class; the rest of the range keeps playing its current strategy. Applies on top of any Overall % lock at this node.';
          lock.addEventListener('click', async () => {
            const freqs = ed.inputs.map(i => Math.max(0, +i.value || 0));
            if (freqs.reduce((a, b) => a + b, 0) <= 0) return toast('set at least one action above 0%', true);
            const edits = combos.map(c => ({ combo: c, freqs }));
            try {
              await api.lock(this.path, { kind: 'hands', edits }, label);
              toast(`${info.label} locked — RE-SOLVE to adapt the tree, or open EXPLOIT to attack it as-is`); this.refresh();
            } catch (e) { toast(e.message, true); }
          });
          foot.appendChild(lock);
        }
      }
      this.appendLockFooter(foot, label, locked);
      body.appendChild(foot);
    }
    el.appendChild(wrap);
  }

  /** Shared "lock as solved" + "unlock" buttons for the lock panel footer. */
  appendLockFooter(foot, label, locked) {
    const freeze = document.createElement('button');
    freeze.className = 'btn ghost'; freeze.textContent = 'LOCK AS SOLVED';
    freeze.dataset.tip = 'The other kind of lock: freeze this node EXACTLY as currently solved \u2014 you change nothing, it just stops adapting. Use it to hold parts of the tree constant while you experiment elsewhere: lock earlier streets as solved, then edit/lock a later node and RE-SOLVE \u2014 only the unlocked parts adapt. (LOCK FREQUENCIES pins YOUR numbers; LOCK AS SOLVED pins the solver\u2019s.)';
    freeze.addEventListener('click', async () => {
      try { await api.lock(this.path, { kind: 'freeze' }, label); toast('locked as solved — this node now stays fixed through re-solves'); this.refresh(); }
      catch (e) { toast(e.message, true); }
    });
    foot.appendChild(freeze);
    if (locked) {
      const un = document.createElement('button');
      un.className = 'btn ghost'; un.textContent = 'UNLOCK';
      un.dataset.tip = 'Remove the lock: this node adapts again on the next RE-SOLVE.';
      un.addEventListener('click', async () => {
        try { await api.unlock(this.path); toast('unlocked'); this.refresh(); }
        catch (e) { toast(e.message, true); }
      });
      foot.appendChild(un);
    }
  }

  async loadRunouts(btn) {
    btn.disabled = true; btn.textContent = 'computing…';
    const forPath = JSON.stringify(this.path); // guard against navigating away mid-compute
    let rep;
    try { rep = await api.runouts(this.path); }
    catch (e) { toast(e.message, true); btn.disabled = false; btn.textContent = 'RUNOUTS REPORT'; return; }
    btn.disabled = false; btn.textContent = 'RUNOUTS REPORT';
    if (JSON.stringify(this.path) !== forPath) return; // stale: user moved on
    this.runoutRep = rep;
    this.runoutSort = { col: 'card', dir: -1 }; // default: high card -> low
    this.renderRunouts();
  }

  renderRunouts() {
    const rep = this.runoutRep;
    const el = this.els.runouts;
    if (!el) return;
    el.innerHTML = '';
    if (!rep) return;
    const colors = computeActionColors(rep.actions, this.view ? this.view.pot : 0);
    // aggression = combined bet/raise frequency; the natural "strategy" sort key
    const aggrIdx = rep.actions
      .map((a, k) => (a.kind === 'bet' || a.kind === 'raise') ? k : -1).filter(k => k >= 0);
    const stratVal = row => {
      if (!row.freqs || !row.freqs.length) return null;          // all-in runout, no action
      if (aggrIdx.length) return aggrIdx.reduce((s, k) => s + (row.freqs[k] || 0), 0);
      return 1 - (row.freqs[0] || 0);                            // fallback: non-first action share
    };
    const valOf = (row, col) =>
      col === 'card' ? RANKS.indexOf(row.card[0]) * 4 + SUITS.indexOf(row.card[1]) :
      col === 'strat' ? stratVal(row) :
      col === 'oopeq' ? row.eq[0] : row.eq[1];

    const { col: sortCol, dir: sortDir } = this.runoutSort;
    const rows = rep.rows.slice().sort((a, b) => {
      const av = valOf(a, sortCol), bv = valOf(b, sortCol);
      if (av == null && bv == null) return 0;
      if (av == null) return 1;                                  // nulls always last
      if (bv == null) return -1;
      return sortDir === -1 ? bv - av : av - bv;
    });

    const arrow = key => key === sortCol ? (sortDir === -1 ? ' ▼' : ' ▲') : '';
    const cls = key => `ro-sort${key === sortCol ? ' sorted' : ''}`;
    const stratLabel = rep.player != null ? this.posLabel(rep.player) + ' strategy' : 'strategy';
    const head = document.createElement('div');
    head.className = 'combo-row head';
    head.innerHTML =
      `<span class="cname ${cls('card')}" data-sort="card" data-tip="Sort by card (rank then suit). Click again to flip direction.">card${arrow('card')}</span>` +
      `<span class="cbar ${cls('strat')}" data-sort="strat" style="background:none" data-tip="Sort by total betting frequency — spot which runouts get barreled most. Click again to flip.">${stratLabel}${arrow('strat')}</span>` +
      `<span class="cnum ${cls('oopeq')}" data-sort="oopeq" data-tip="Sort by OOP equity. Click again to flip.">OOP eq${arrow('oopeq')}</span>` +
      `<span class="cnum ${cls('ipeq')}" data-sort="ipeq" data-tip="Sort by IP equity. Click again to flip.">IP eq${arrow('ipeq')}</span>`;
    el.appendChild(head);

    for (const row of rows) {
      const r = document.createElement('div');
      r.className = 'combo-row';
      const name = `<span class="suit-${row.card[1]}">${row.card[0]}${SUIT_GLYPH[row.card[1]]}</span>`;
      const bar = row.freqs.map((f, k) =>
        `<div style="width:${(f * 100).toFixed(1)}%;background:${colors[k]}" data-tip="${rep.actions[k].label}: ${(f * 100).toFixed(1)}% of range"></div>`).join('');
      r.innerHTML = `<span class="cname">${name}</span><span class="cbar">${bar}</span>
        <span class="cnum">${row.eq[0] != null ? (row.eq[0] * 100).toFixed(1) : '—'}</span>
        <span class="cnum">${row.eq[1] != null ? (row.eq[1] * 100).toFixed(1) : '—'}</span>`;
      el.appendChild(r);
    }
    // legend
    const leg = document.createElement('div');
    leg.className = 'legend';
    rep.actions.forEach((a, k) => {
      leg.innerHTML += `<span class="key"><i style="background:${colors[k]}"></i>${a.label}</span>`;
    });
    el.appendChild(leg);

    head.querySelectorAll('.ro-sort').forEach(h =>
      h.addEventListener('click', () => {
        const col = h.dataset.sort;
        if (this.runoutSort.col === col) this.runoutSort.dir *= -1; // same column: flip
        else this.runoutSort = { col, dir: -1 };                    // new column: desc first
        this.renderRunouts();
      }));
  }

  // ----- matrix -----

  // ----- filters -----

  filtersActive() {
    return this.filterCats.size > 0 || this.filterSuits.size > 0;
  }

  /** Hover-preview: dim matrix cells with no combos matching the hovered filter. */
  applyFilterPreview() {
    const pv = this.filterPreview;
    const p = this.player;
    for (let i = 0; i < 13; i++) {
      for (let j = 0; j < 13; j++) {
        const cell = this.cells[i * 13 + j];
        if (!pv) { cell.classList.remove('pdim'); continue; }
        let hasMatch = false;
        for (const [a, b] of cellCombos(cellInfo(i, j))) {
          const hi = this.handIdx[p].get(comboIndex(a, b));
          if (hi === undefined) continue;
          const h = this.view.players[p].hands[hi];
          if (h.reach <= 1e-9) continue;
          const c = this.cats[p][hi];
          const m = pv.type === 'suit'
            ? c.suits.has(pv.key)
            : (c.made === pv.key || c.draw === pv.key || c.eqs === pv.key || c.eqa === pv.key);
          if (m) { hasMatch = true; break; }
        }
        cell.classList.toggle('pdim', !hasMatch);
      }
    }
  }

  setFilterPreview(pv) {
    this.filterPreview = pv;
    this.applyFilterPreview();
  }

  /** Does hand index hi of player p pass the active filters? */
  handMatches(p, hi) {
    if (!this.filtersActive()) return true;
    const c = this.cats[p][hi];
    let match = true;
    if (this.filterCats.size) {
      match = this.filterCats.has(c.made) || this.filterCats.has(c.draw)
        || this.filterCats.has(c.eqs) || this.filterCats.has(c.eqa);
    }
    if (match && this.filterSuits.size) {
      match = [...c.suits].some(t => this.filterSuits.has(t));
    }
    return this.filterMode === 'include' ? match : !match;
  }

  // ----- Exploit (max-exploit / best-response) mode -----

  exploitReady() {
    return !!(this.exploit && this.exploit.exploiter === this.player
      && this.exploit._path === JSON.stringify(this.path));
  }

  /** Effective render mode: in EXPLOIT, strategy-style cells when the
   *  exploiter acts here, else an EV-style heatmap of the per-hand gain. */
  effMode() {
    if (this.mode !== 'exploit') return this.mode;
    return this.exploitReady() && this.exploit.player === this.exploit.exploiter
      ? 'strategy' : 'ev';
  }

  /** Hand array driving the matrix: the exploit overlay (strategy = BR
   *  actions, ev = gain vs current) when active, else the solver view. */
  handsFor(p) {
    if (this.mode === 'exploit' && this.exploitReady() && p === this.player) {
      if (!this._exploitHands) {
        this._exploitHands = this.exploit.hands.map(h => ({
          ...h,
          strategy: h.br_strategy || null,
          ev: h.gain != null ? h.gain : null,
          eq: null,
          evs: h.evs || null,
        }));
      }
      return this._exploitHands;
    }
    return this.view.players[p].hands;
  }

  async loadExploit() {
    if (this._exploitLoading) return;
    this._exploitLoading = true;
    const forPath = JSON.stringify(this.path), forPlayer = this.player;
    try {
      const ex = await api.exploit(this.path, this.player);
      if (JSON.stringify(this.path) === forPath && this.player === forPlayer
          && this.mode === 'exploit') {
        ex._path = forPath;
        this.exploit = ex;
        this._exploitHands = null;
        this.renderActions();
        this.renderMatrix();
        this.renderLegend();
      }
    } catch (e) {
      toast(e.message, true);
      this.els.matrix.style.opacity = ''; // un-dim; grid keeps its last paint
    }
    finally { this._exploitLoading = false; }
  }

  renderExploitBanner(el) {
    const b = document.createElement('div');
    b.className = 'exploit-banner';
    if (!this.exploitReady()) {
      b.innerHTML = `<b>EXPLOIT</b> computing best response for ${this.posLabel(this.player)}…`;
    } else {
      const e = this.exploit;
      const nm = this.posLabel(e.exploiter);
      b.innerHTML = e.avg_gain != null
        ? `<b>EXPLOIT</b> ${nm} best response vs current strategy: EV ` +
          `<b>${fmt(e.avg_br_ev)}</b> vs ${fmt(e.avg_cur_ev)} → gains ` +
          `<b>${fmt(e.avg_gain)}</b> chips/hand (reach-weighted)` +
          (e.locked ? ' · this node is locked: BR cannot deviate here' : '')
        : `<b>EXPLOIT</b> ${nm} has no reachable hands at this node.`;
    }
    el.appendChild(b);
  }

  /** Per-combo data for a cell (present + filter-matching, with reach): for the
   *  per-combo EV/EQ columns. */
  cellCombosData(i, j, p) {
    const idx = this.handIdx[p];
    const hands = this.handsFor(p);
    const out = [];
    for (const [a, b] of cellCombos(cellInfo(i, j))) {
      const hi = idx.get(comboIndex(a, b));
      if (hi === undefined || !this.handMatches(p, hi)) continue;
      const h = hands[hi];
      if (h.reach <= 1e-9) continue;
      out.push({ reach: h.reach, ev: h.ev, eq: h.eq, strat: h.strategy,
                 combo: cardToString(a) + cardToString(b) });
    }
    return out;
  }

  cellAgg(i, j, p) {
    // Aggregate hands of player p within a cell class.
    const combos = cellCombos(cellInfo(i, j));
    const hands = this.handsFor(p);
    const idx = this.handIdx[p];
    let reach = 0, weight = 0, ev = 0, evW = 0, eq = 0, eqW = 0;
    let strat = null, na = 0;
    if (this.view.node_type === 'action' && this.view.player === p) {
      na = this.view.actions.length;
      strat = new Array(na).fill(0);
    }
    let present = 0, possible = 0;
    for (const [a, b] of combos) {
      // height is normalized over combos the board actually allows, not the
      // theoretical 4/6/12 (a board card removes that suit's combo)
      if (this.boardSet && (this.boardSet.has(a) || this.boardSet.has(b))) continue;
      possible++;
      const hi = idx.get(comboIndex(a, b));
      if (hi === undefined) continue;
      if (!this.handMatches(p, hi)) continue;
      const h = hands[hi];
      present++;
      reach += h.reach;
      weight += h.weight;
      if (h.ev != null) { ev += h.ev * h.reach; evW += h.reach; }
      if (h.eq != null) { eq += h.eq * h.reach; eqW += h.reach; }
      if (strat && h.strategy) h.strategy.forEach((s, k) => strat[k] += s * h.reach);
    }
    if (strat && reach > 1e-12) strat = strat.map(s => s / reach);
    return { present, total: possible, reach, weight,
             ev: evW > 1e-12 ? ev / evW : null,
             eq: eqW > 1e-12 ? eq / eqW : null, strat };
  }

  renderMatrix() {
    const p = this.player;
    const colors = this.actionColors();
    if (this.mode === 'exploit' && !this.exploitReady()) {
      // Don't paint a misleading frame of non-exploit colors while the BR
      // data loads — keep the previous grid, dimmed; loadExploit re-renders.
      this.loadExploit();
      this.els.matrix.style.opacity = '.45';
      return;
    }
    this.els.matrix.style.opacity = '';
    const effMode = this.effMode();
    const hands = this.handsFor(p);
    const aggs = [];
    for (let i = 0; i < 13; i++)
      for (let j = 0; j < 13; j++) aggs.push(this.cellAgg(i, j, p));
    // Normalize fill height to the busiest hand class in the FULL (unfiltered)
    // range, so applying a filter just blanks non-matching cells instead of
    // rescaling the height of the ones that remain.
    let maxReach = 1e-12;
    for (let i = 0; i < 13; i++) {
      for (let j = 0; j < 13; j++) {
        const combos = cellCombos(cellInfo(i, j));
        let r = 0, poss = 0;
        for (const [a, b] of combos) {
          if (this.boardSet && (this.boardSet.has(a) || this.boardSet.has(b))) continue;
          poss++;
          const hi = this.handIdx[p].get(comboIndex(a, b));
          if (hi !== undefined) r += hands[hi].reach;
        }
        maxReach = Math.max(maxReach, r / Math.max(poss, 1));
      }
    }
    // Fill mode: how much of a cell/column to fill for a given per-combo reach.
    //  normalized -> relative to the busiest hand class (best use of space)
    //  range      -> the hand's actual weight/reach out of a full combo
    //  full       -> fill completely (ignore reach), just show the colors
    const fillMode = this.fillMode || 'normalized';
    const fillH = (perCombo) => {
      if (perCombo <= 1e-9) return 0;
      if (fillMode === 'full') return 1;
      if (fillMode === 'range') return Math.min(1, perCombo);
      return Math.min(1, perCombo / maxReach);
    };
    // EV colour ramp range across the displayed (filtered) combos
    let evMin = Infinity, evMax = -Infinity;
    if (effMode === 'ev') {
      this.handsFor(p).forEach((h, hi) => {
        if (h.reach > 1e-9 && h.ev != null && this.handMatches(p, hi)) {
          if (h.ev < evMin) evMin = h.ev;
          if (h.ev > evMax) evMax = h.ev;
        }
      });
    }
    const evSpan = evMax - evMin;
    const evColor = (ev) => {
      const t = evSpan > 1e-6 ? Math.max(0, Math.min(1, (ev - evMin) / evSpan)) : 0.5;
      return `hsl(${(t * 130).toFixed(0)} 70% 50%)`;
    };
    const eqColor = (eq) => `hsl(${(eq * 130).toFixed(0)} 75% 52%)`;
    // render each combo as its own bottom-anchored vertical column
    const renderCols = (bars, fill, sub, list, colorFn, aggText) => {
      fill.style.opacity = 0;
      if (!list.length) { bars.style.opacity = 0; sub.textContent = ''; return; }
      bars.style.opacity = 1;
      bars.style.height = '100%';
      bars.style.alignItems = 'flex-end';
      const w = (100 / list.length).toFixed(3);
      bars.innerHTML = list.map(c =>
        `<div data-combo="${c.combo}" style="width:${w}%;height:${(fillH(c.reach) * 100).toFixed(1)}%;background:${colorFn(c)}"></div>`
      ).join('');
      sub.textContent = aggText;
    };
    // combined (Horizontal) value cell: one bottom-anchored bar coloured by the
    // cell's aggregate EV/EQ, height = how much of the cell reaches the node
    // (mirrors the STRAT combined bar via the same `intensity`).
    const renderAgg = (bars, fill, sub, color, intensity, aggText, reach) => {
      bars.style.opacity = 0;
      const show = color != null && reach > 1e-9;
      fill.style.height = `${(intensity * 100).toFixed(1)}%`;
      fill.style.background = color || 'transparent';
      fill.style.opacity = show ? 1 : 0;
      sub.textContent = aggText;
    };
    for (let i = 0; i < 13; i++) {
      for (let j = 0; j < 13; j++) {
        const cell = this.cells[i * 13 + j];
        const agg = aggs[i * 13 + j];
        const bars = cell.querySelector('.bars');
        const fill = cell.querySelector('.fill');
        const sub = cell.querySelector('.sub');
        bars.innerHTML = '';
        bars.style.height = '';
        bars.style.alignItems = '';
        bars.style.flexDirection = '';
        bars.style.gap = '';
        fill.style.height = '0';
        cell.classList.toggle('empty', agg.present === 0 || agg.reach <= 1e-9);
        cell.classList.toggle('absent', agg.present === 0);
        cell.classList.toggle('selected',
          this.selectedCell && this.selectedCell[0] === i && this.selectedCell[1] === j);
        const intensity = fillH(agg.reach / Math.max(agg.total, 1));
        if (effMode === 'strategy' && agg.strat
            && this.actionFilter != null && this.actionFilter < agg.strat.length) {
          // Action filter: show ONLY the selected action, in its colour. The
          // fill mode AND orientation apply exactly like the unfiltered STRAT
          // view -- the bar is just that action's slice of the hand's reach-fill
          // (Full -> pure frequency, Range/Normalized -> reach-weighted).
          const af = this.actionFilter;
          if (this.comboRows) {
            // Vertical: one column per combo, height = reach-fill x this combo's
            // frequency for the selected action.
            const list = this.cellCombosData(i, j, p).filter(c => c.strat);
            if (!list.length) { bars.style.opacity = 0; sub.textContent = ''; }
            else {
              bars.style.opacity = 1;
              bars.style.height = '100%';
              bars.style.alignItems = 'flex-end';
              const w = (100 / list.length).toFixed(3);
              bars.innerHTML = list.map(c => {
                const colH = fillH(c.reach) * (c.strat[af] || 0) * 100;
                return `<div data-combo="${c.combo}" style="width:${w}%;height:${colH.toFixed(1)}%;background:${colors[af]}"></div>`;
              }).join('');
              sub.textContent = '';
            }
          } else {
            // Horizontal: single bar, height = reach-fill x action frequency.
            const f = agg.strat[af];
            const d = document.createElement('div');
            d.style.width = '100%';
            d.style.background = colors[af];
            bars.appendChild(d);
            const show = agg.reach > 1e-9 && f > 0.002;
            bars.style.opacity = show ? 1 : 0;
            bars.style.height = `${(intensity * f * 100).toFixed(1)}%`;
            sub.textContent = show ? `${Math.round(f * 100)}%` : '';
          }
        } else if (effMode === 'strategy' && agg.strat && this.comboRows) {
          // one vertical column per combo (cell chopped vertically): column
          // height = its reach, split by that combo's action frequencies with
          // the most aggressive actions (reds) at the bottom
          const list = this.cellCombosData(i, j, p).filter(c => c.strat);
          if (!list.length) { bars.style.opacity = 0; sub.textContent = ''; }
          else {
            bars.style.opacity = 1;
            bars.style.height = '100%';
            bars.style.alignItems = 'flex-end'; // bottom-anchor the columns
            const w = (100 / list.length).toFixed(3);
            bars.innerHTML = list.map(c => {
              const colH = fillH(c.reach) * 100;
              const segs = c.strat.map((s, k) =>
                `<div style="height:${(s * 100).toFixed(2)}%;background:${colors[k]}"></div>`).join('');
              return `<div data-combo="${c.combo}" style="width:${w}%;height:${colH.toFixed(1)}%;display:flex;flex-direction:column">${segs}</div>`;
            }).join('');
            sub.textContent = '';
          }
        } else if (effMode === 'strategy' && agg.strat) {
          agg.strat.forEach((s, k) => {
            const d = document.createElement('div');
            d.style.width = `${s * 100}%`;
            d.style.background = colors[k];
            bars.appendChild(d);
          });
          // Discrete action colors at full opacity. How much of this hand
          // reaches the node is shown as vertical fill height ("Range
          // style), bottom-anchored — not by dimming the colors.
          bars.style.opacity = agg.reach > 1e-9 ? 1 : 0;
          bars.style.height = `${(intensity * 100).toFixed(1)}%`;
          // "Strategy + EV" display: the hand's EV in the cell corner
          sub.textContent =
            agg.ev != null && agg.reach > 1e-9 ? fmt(agg.ev) : '';
        } else if (effMode === 'strategy') {
          // not the actor: range-weight presence as a vertical orange bar
          bars.style.opacity = 0;
          fill.style.height = `${(intensity * 100).toFixed(1)}%`;
          fill.style.background = '#f28c26';
          fill.style.opacity = agg.reach > 1e-9 ? 0.9 : 0;
          sub.textContent = '';
        } else if (effMode === 'ev') {
          // Vertical (comboRows): one column per combo, height = its reach,
          // colour = its EV, natural combo order (same column = same combo as
          // STRAT/EQ). Horizontal: one combined bar coloured by aggregate EV.
          const evText = agg.ev != null && agg.reach > 1e-9 ? fmt(agg.ev) : '';
          if (this.comboRows) {
            const list = this.cellCombosData(i, j, p).filter(c => c.ev != null);
            renderCols(bars, fill, sub, list, c => evColor(c.ev), evText);
          } else {
            renderAgg(bars, fill, sub, agg.ev != null ? evColor(agg.ev) : null,
              intensity, evText, agg.reach);
          }
        } else if (effMode === 'eq') {
          const eqText = agg.eq != null && agg.reach > 1e-9 ? `${Math.round(agg.eq * 100)}%` : '';
          if (this.comboRows) {
            const list = this.cellCombosData(i, j, p).filter(c => c.eq != null);
            renderCols(bars, fill, sub, list, c => eqColor(c.eq), eqText);
          } else {
            renderAgg(bars, fill, sub, agg.eq != null ? eqColor(agg.eq) : null,
              intensity, eqText, agg.reach);
          }
        }
      }
    }
  }

  renderLegend() {
    const el = this.els.legend;
    el.innerHTML = '';
    if (this.mode === 'exploit') {
      const key = document.createElement('span');
      key.className = 'key';
      key.textContent = this.effMode() === 'strategy'
        ? `EXPLOIT: cells = ${this.posLabel(this.player)}'s best-response actions (mostly pure); cell number = EV gain vs current strategy`
        : `EXPLOIT: heatmap = ${this.posLabel(this.player)}'s per-hand EV gain from best-responding vs the current strategy`;
      el.appendChild(key);
      return;
    }
    if (this.view.node_type === 'action' && this.mode === 'strategy'
        && this.view.player === this.player) {
      const colors = this.actionColors();
      if (this.actionFilter != null && this.actionFilter < this.view.actions.length) {
        const k = this.actionFilter;
        const key = document.createElement('span');
        key.className = 'key';
        key.innerHTML = `<i style="background:${colors[k]}"></i>${this.view.actions[k].label} frequency (bar height = how often each hand takes it)`;
        el.appendChild(key);
      } else {
        this.view.actions.forEach((a, k) => {
          const key = document.createElement('span');
          key.className = 'key';
          key.innerHTML = `<i style="background:${colors[k]}"></i>${a.label}`;
          el.appendChild(key);
        });
      }
    } else if (this.mode === 'ev') {
      el.innerHTML = `<span class="key"><i style="background:hsl(0 70% 50%)"></i>lowest EV</span>
                      <span class="key"><i style="background:hsl(65 70% 50%)"></i>mid</span>
                      <span class="key"><i style="background:hsl(130 70% 50%)"></i>highest EV</span>
                      <span class="key dim">· each column = one combo</span>`;
    } else if (this.mode === 'eq') {
      el.innerHTML = `<span class="key"><i style="background:hsl(0 75% 52%)"></i>0%</span>
                      <span class="key"><i style="background:hsl(65 75% 52%)"></i>50%</span>
                      <span class="key"><i style="background:hsl(130 75% 52%)"></i>100%</span>`;
    } else {
      el.innerHTML = `<span class="key"><i style="background:#f28c26"></i>range weight</span>`;
    }
  }

  // ----- hands panel (hover-driven) -----

  selectCell(i, j) {
    // click pins/unpins a cell so it stays after the mouse leaves the matrix
    if (this.selectedCell && this.selectedCell[0] === i && this.selectedCell[1] === j) {
      this.selectedCell = null;
    } else {
      this.selectedCell = [i, j];
    }
    this.renderMatrix();
    this.renderHandsPanel();
    // per-hand lock editor targets the pinned cell — refresh it on pin change
    if (this.lockMode === 'hands' && this.view && this.view.node_type === 'action') {
      this.renderActions();
    }
  }

  /** Villain-profile locks: when the spot came from the Preflop Lab with a
   *  modeled seat, one click compiles that player's postflop stats into node
   *  locks across his whole tree (equilibrium distortion via Range rakes). */
  renderVillainLocks() {
    const box = document.getElementById('villain-locks');
    if (!box) return;
    box.innerHTML = '';
    const pf = this.preflop;
    if (!pf || !pf.villains) return;
    for (const [side, p] of [['oop', 0], ['ip', 1]]) {
      const v = pf.villains[side];
      if (!v) continue;
      const on = this.villainLocked === p;
      const b = document.createElement('button');
      b.className = 'btn ghost sm vlock' + (on ? ' on' : '');
      b.textContent = on
        ? `\u{1F512} ${this.posLabel(p)} = ${v.name} \u00b7 unlock`
        : `LOCK ${this.posLabel(p)} TO ${v.name}`;
      b.dataset.tip = on
        ? `${v.name}'s postflop stats are locked into every ${this.posLabel(p)} decision. RE-SOLVE adapts your play; EXPLOIT reads the max punishment. Click to clear.`
        : `Compile ${v.name}'s postflop stats (c-bet ${v.stats.cbet.join('/')} \u00b7 fold-to-bet ${v.stats.fold_to_bet.join('/')} \u00b7 raise ${v.stats.raise_bet}%) into locks on every ${this.posLabel(p)} decision \u2014 his natural betting hands keep betting, raked to the stat targets. Then RE-SOLVE or EXPLOIT.`;
      b.addEventListener('click', () => this.toggleVillainLock(p, v));
      box.appendChild(b);
    }
  }

  async toggleVillainLock(p, v) {
    try {
      if (this.villainLocked === p) {
        await api.profileLocksClear();
        this.villainLocked = null;
        toast('villain profile locks cleared');
      } else {
        const out = await api.profileLocks(p, v.stats,
          this.preflop ? this.preflop.aggressor : null);
        this.villainLocked = p;
        const worst = (out.rows || []).reduce((w, r) =>
          !w || Math.abs(r.achieved - r.target) > Math.abs(w.achieved - w.target) ? r : w, null);
        toast(`${out.locked} nodes locked to ${v.name}` +
          (worst ? ` (worst fit: ${worst.label} ${worst.achieved.toFixed(0)}% vs target ${worst.target.toFixed(0)}%)` : '') +
          ' \u2014 RE-SOLVE to adapt, then EXPLOIT');
      }
      await this.refresh(); // lock badges + strategies update everywhere
    } catch (e) { toast(e.message, true); }
  }

  renderHandsPanel() {
    const content = this.els.handsContent;
    const label = this.els.handsLabel;
    if (!content) return;
    if (!this.view) { content.innerHTML = ''; label.textContent = ''; return; }
    // Hovering a hand temporarily flips FILTERS/BLOCKERS to the HANDS view, so
    // you can inspect a combo without leaving your filter; it reverts on exit.
    const flip = this.hoverCell && (this.handsTab === 'filters' || this.handsTab === 'blockers');
    const eff = flip ? 'hands' : this.handsTab;
    if (this.els.handsTabs) {
      this.els.handsTabs.querySelectorAll('.htab').forEach(x =>
        x.classList.toggle('active', x.dataset.v === eff));
    }
    if (eff === 'filters') {
      const n = this.filterCats.size + this.filterSuits.size;
      label.textContent = n ? `${n} active · ${this.filterMode}` : '';
      this.renderFiltersPanel(content);
      return;
    }
    if (eff === 'blockers') {
      label.textContent = '';
      this.renderBlockersPanel(content);
      return;
    }
    const ref = this.hoverCell || this.selectedCell;
    if (!ref) {
      label.textContent = '';
      content.innerHTML = '<div class="dim" style="padding:10px 4px;font-size:11px">hover a hand in the matrix — click to pin</div>';
      return;
    }
    const [i, j] = ref;
    const info = cellInfo(i, j);
    const pinned = this.selectedCell && !this.hoverCell;
    label.textContent = info.label + (pinned ? ' · pinned' : '');

    const d = this.cellHandData(i, j);
    if (!d.cand.length) {
      content.innerHTML = '<div class="dim" style="padding:10px 4px;font-size:11px">not in range</div>';
      return;
    }
    if (eff === 'hands') {
      content.innerHTML = this.handTilesHtml(d);
    } else {
      content.innerHTML = this.summaryHtml(
        d.cand.filter(x => x[3]), d.isActor, d.colors, d.acts);
    }
  }

  // Shared by the HANDS tab and the grid-hover popup: gather a cell's
  // in-deck combos. ALL of them are shown (dropping combos would skew the
  // SUMMARY stats); combos that barely reach this
  // node — e.g. a suit that folded out on an earlier street keeps a tiny
  // non-zero reach — are dimmed rather than hidden, judged relative to the
  // busiest combo of the SAME hand.
  cellHandData(i, j) {
    const info = cellInfo(i, j);
    const p = this.player;
    const hands = this.view.players[p].hands;
    const idx = this.handIdx[p];
    const cand = [];
    let cellMaxReach = 0;
    for (const [a, b] of cellCombos(info)) {
      const hi = idx.get(comboIndex(a, b));
      if (hi === undefined) continue;
      const h = hands[hi];
      if (h.reach > cellMaxReach) cellMaxReach = h.reach;
      cand.push([h, a, b, this.handMatches(p, hi)]);
    }
    return {
      cand,
      minReach: Math.max(1e-9, cellMaxReach * 0.01),
      isActor: this.view.node_type === 'action' && this.view.player === p,
      colors: this.actionColors(),
      acts: this.view.actions || [],
    };
  }

  handTilesHtml({ cand, minReach, isActor, colors, acts }) {
    // Layout: 2 columns up to 4 combos, 3 columns beyond.
    const cols3 = cand.length > 4;
    const tiles = cand.map(([h, a, b, m]) =>
      this.handTile(h, a, b, isActor, colors, acts, cols3,
        !m || h.reach < minReach));
    return `<div class="hand-tiles${cols3 ? ' cols3' : ''}">${tiles.join('')}</div>`;
  }

  // ----- grid-hover popup: the HANDS view in miniature -----

  showHandPop(i, j, cellEl) {
    if (!this.view) return;
    if (!this.handPop) {
      this.handPop = document.createElement('div');
      this.handPop.id = 'hand-pop';
      document.body.appendChild(this.handPop);
    }
    const d = this.cellHandData(i, j);
    if (!d.cand.length) return this.hideHandPop();
    const pop = this.handPop;
    pop.innerHTML = this.handTilesHtml(d);
    pop.classList.remove('hidden');
    // beside the hovered cell, flipped/clamped to stay on screen
    const r = cellEl.getBoundingClientRect();
    let x = r.right + 10;
    if (x + pop.offsetWidth > window.innerWidth - 8) x = r.left - pop.offsetWidth - 10;
    const y = Math.max(8, Math.min(r.top - 8, window.innerHeight - pop.offsetHeight - 8));
    pop.style.left = `${Math.max(8, x)}px`;
    pop.style.top = `${y}px`;
  }

  hideHandPop() {
    if (this.handPop) this.handPop.classList.add('hidden');
  }

  handTile(h, a, b, isActor, colors, acts, compact, dimmed) {
    const name =
      `<span class="suit-${SUITS[suit(a)]}">${RANKS[rank(a)]}${SUIT_GLYPH[SUITS[suit(a)]]}</span>` +
      `<span class="suit-${SUITS[suit(b)]}">${RANKS[rank(b)]}${SUIT_GLYPH[SUITS[suit(b)]]}</span>`;
    const meta = `${h.eq != null ? (h.eq * 100).toFixed(0) + '% eq · ' : ''}EV`;
    let body;
    if (isActor && h.strategy) {
      // Horizontal split, aggressive actions on the left;
      // action labels with EVs overlaid bottom-left/right.
      const segs = [];
      const lines = [];
      for (let k = acts.length - 1; k >= 0; k--) {
        const f = h.strategy[k];
        if (f < 0.001) continue;
        const ev = h.evs && h.evs[k] != null ? fmt(h.evs[k]) : '—';
        segs.push(
          `<div style="flex:${f.toFixed(4)};background:${colors[k]}" ` +
          `data-tip="${acts[k].label}: ${(f * 100).toFixed(1)}% of the time · EV ${ev}"></div>`);
        lines.push(
          `<div class="hand-line"><span class="hl-lab">${acts[k].label}</span>` +
          `<span class="hl-ev">${ev}</span></div>`);
      }
      body = `<div class="htb-h${compact ? ' short' : ''}">` +
        `<div class="hseg-row">${segs.join('')}</div>` +
        `<div class="hand-lines">${lines.join('')}</div></div>`;
    } else {
      body = `<div class="htb flat"><span>EV <b>${h.ev != null ? fmt(h.ev) : '—'}</b></span>` +
        `<span>EQ <b>${h.eq != null ? (h.eq * 100).toFixed(1) + '%' : '—'}</b></span></div>`;
    }
    return `<div class="hand-tile${dimmed ? ' fdim' : ''}"><div class="hth"><span>${name}</span><span class="meta">${meta}</span></div>${body}</div>`;
  }

  // ----- Filters tab -----

  renderFiltersPanel(content) {
    const p = this.player;
    const hands = this.view.players[p].hands;
    const cats = this.cats[p];
    const isActor = this.view.node_type === 'action' && this.view.player === p;
    const colors = this.actionColors();
    const na = (this.view.actions || []).length;

    // reach + strategy mix per category
    const agg = new Map();
    let totalReach = 0;
    hands.forEach((h, i) => {
      totalReach += h.reach;
      for (const key of [cats[i].made, cats[i].draw, cats[i].eqs, cats[i].eqa]) {
        if (!key) continue;
        let a = agg.get(key);
        if (!a) { a = { reach: 0, strat: new Array(na).fill(0) }; agg.set(key, a); }
        a.reach += h.reach;
        if (isActor && h.strategy) h.strategy.forEach((s, k) => a.strat[k] += s * h.reach);
      }
    });

    const row = (key, label) => {
      const a = agg.get(key);
      const sel = this.filterCats.has(key);
      if ((!a || a.reach <= 1e-12) && !sel) return '';
      const pct = a && totalReach > 1e-12 ? (a.reach / totalReach) * 100 : 0;
      let bar = '';
      if (isActor && a && a.reach > 1e-12) {
        bar = a.strat.map((s, k) =>
          `<div style="width:${((s / a.reach) * 100).toFixed(1)}%;background:${colors[k]}"></div>`).join('');
      }
      return `<div class="filter-row${sel ? ' sel' : ''}" data-key="${key}"
        data-tip="Click to filter the matrix by this category. The bar shows how these hands play here.">
        <span class="fname">${label}</span>
        <span class="fpct">${pct.toFixed(1)}%</span>
        <span class="fbar">${bar}</span></div>`;
    };

    const section = (title, entries) => {
      const rows = entries.map(([k, l]) => row(k, l)).filter(Boolean).join('');
      return rows ? `<div class="filter-sec"><div class="fsec-title">${title}</div>${rows}</div>` : '';
    };

    const suitBtn = (tag, html, tip) =>
      `<button class="suit-btn${this.filterSuits.has(tag) ? ' sel' : ''}" data-suit="${tag}" data-tip="${tip}">${html}</button>`;
    const offsuitBtns = [3, 2, 1, 0].map(s =>
      suitBtn(`o${s}`, `<span class="suit-${SUITS[s]}">${SUIT_GLYPH[SUITS[s]]}</span>`,
        'Offsuit and pocket-pair combos containing this suit.')).join('');
    const suitedBtns = [3, 2, 1, 0].map(s =>
      suitBtn(`s${s}`, `<span class="suit-${SUITS[s]}">${SUIT_GLYPH[SUITS[s]]}${SUIT_GLYPH[SUITS[s]]}</span>`,
        'Suited combos of this suit.')).join('');

    content.innerHTML = `
      <div class="filter-top">
        <div class="seg">
          <button data-fmode="include" class="${this.filterMode === 'include' ? 'active' : ''}"
            data-tip="Matrix shows ONLY hands matching the selected filters.">INCLUDE</button>
          <button data-fmode="exclude" class="${this.filterMode === 'exclude' ? 'active' : ''}"
            data-tip="Matrix hides hands matching the selected filters.">EXCLUDE</button>
        </div>
        <button class="btn ghost" id="filter-clear" data-tip="Clear all active filters.">clear</button>
      </div>
      <div class="filter-suits">
        <span class="dim">Offsuit</span>${offsuitBtns}
        <span class="dim" style="margin-left:12px">Suited</span>${suitedBtns}
      </div>
      <div class="filter-cols">
        <div>
          ${section('Hands', MADE_ORDER.map(k => [k, MADE_LABELS[k]]))}
          ${section('EQ buckets — simple', Object.entries(EQS_LABELS))}
        </div>
        <div>
          ${this.view.board.length < 5 ? section('Draws', DRAW_ORDER.map(k => [k, DRAW_LABELS[k]])) : ''}
          ${section('EQ buckets — advanced', Object.entries(EQA_LABELS))}
        </div>
      </div>`;

    content.querySelectorAll('.filter-row').forEach(r => {
      r.addEventListener('click', () => {
        const k = r.dataset.key;
        if (this.filterCats.has(k)) this.filterCats.delete(k);
        else this.filterCats.add(k);
        this.applyFilters();
      });
      // hover preview: highlight matching hands in the matrix
      r.addEventListener('mouseenter', () =>
        this.setFilterPreview({ type: 'cat', key: r.dataset.key }));
      r.addEventListener('mouseleave', () => this.setFilterPreview(null));
    });
    content.querySelectorAll('.suit-btn').forEach(b => {
      b.addEventListener('click', () => {
        const t = b.dataset.suit;
        if (this.filterSuits.has(t)) this.filterSuits.delete(t);
        else this.filterSuits.add(t);
        this.applyFilters();
      });
      b.addEventListener('mouseenter', () =>
        this.setFilterPreview({ type: 'suit', key: b.dataset.suit }));
      b.addEventListener('mouseleave', () => this.setFilterPreview(null));
    });
    content.querySelectorAll('[data-fmode]').forEach(b =>
      b.addEventListener('click', () => {
        this.filterMode = b.dataset.fmode;
        this.applyFilters();
      }));
    content.querySelector('#filter-clear').addEventListener('click', () => {
      this.filterCats.clear();
      this.filterSuits.clear();
      this.applyFilters();
    });
    // panel was rebuilt: re-sync the preview dim (hovered row may be gone)
    this.applyFilterPreview();
  }

  applyFilters() {
    this.renderMatrix();
    this.renderHandsPanel();
  }


  // ----- equity distribution chart -----

  buildEqCurves() {
    this.eqCurves = [0, 1].map(p => {
      const hands = this.view.players[p].hands;
      const items = hands
        .map((h, i) => ({ i, eq: h.eq, reach: h.reach }))
        .filter(x => x.eq != null && x.reach > 1e-9);
      items.sort((a, b) => a.eq - b.eq);
      const total = items.reduce((s, x) => s + x.reach, 0);
      let acc = 0;
      const pts = [];
      const posByHand = new Map();
      for (const x of items) {
        const xc = total > 0 ? (acc + x.reach / 2) / total : 0;
        acc += x.reach;
        pts.push({ x: xc, y: x.eq, i: x.i });
        posByHand.set(x.i, { x: xc, y: x.eq });
      }
      return { pts, posByHand };
    });
  }

  renderEqStats() {
    const el = this.els.eqStats;
    if (!el) return;
    const pot = this.view.pot;
    el.innerHTML = [0, 1].map(p => {
      const hands = this.view.players[p].hands;
      let r = 0, ev = 0, evW = 0, eq = 0, eqW = 0;
      hands.forEach(h => {
        r += h.reach;
        if (h.ev != null) { ev += h.ev * h.reach; evW += h.reach; }
        if (h.eq != null) { eq += h.eq * h.reach; eqW += h.reach; }
      });
      const avgEv = evW > 1e-12 ? ev / evW : null;
      const avgEq = eqW > 1e-12 ? eq / eqW : null;
      const eqr = avgEv != null && avgEq > 1e-9 ? avgEv / (avgEq * pot) : null;
      return `<div class="eqstat" data-tip="${this.posLabel(p)} at this node. EQR = equity realization: EV as a fraction of (equity × pot). Under 100% means this range under-realizes its equity — typical for out-of-position or capped ranges.">
        <div class="eqstat-head"><i style="background:${EQ_COLORS[p]}"></i>${this.posLabel(p)}</div>
        <div class="eqstat-grid">
          <span><label>EV</label><div>${avgEv != null ? fmt(avgEv) : '—'}</div></span>
          <span><label>Equity</label><div>${avgEq != null ? (avgEq * 100).toFixed(1) + '%' : '—'}</div></span>
          <span><label>EQR</label><div>${eqr != null ? (eqr * 100).toFixed(0) + '%' : '—'}</div></span>
          <span><label>Combos</label><div>${r.toFixed(1)}</div></span>
        </div></div>`;
    }).join('');
  }

  drawEquityChart() {
    const cv = this.els.eqCanvas;
    if (!cv || !this.view || !this.eqCurves) return;
    // render at the element's CSS size x devicePixelRatio so text stays crisp
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
    const { L, R, T, B } = EQ_M;
    const px = x => L + x * (W - L - R);
    const py = y => H - B - y * (H - T - B);
    ctx.clearRect(0, 0, W, H);

    // grid
    ctx.strokeStyle = '#262626';
    ctx.fillStyle = '#8a8a8a';
    ctx.font = '12px IBM Plex Mono';
    for (const v of [0.25, 0.5, 0.75]) {
      ctx.beginPath(); ctx.moveTo(px(0), py(v)); ctx.lineTo(px(1), py(v)); ctx.stroke();
      ctx.fillText(`${v * 100}`, 8, py(v) + 4);
      ctx.beginPath(); ctx.moveTo(px(v), py(0)); ctx.lineTo(px(v), py(1)); ctx.stroke();
      ctx.fillText(`${v * 100}`, px(v) - 9, H - 7);
    }

    // curves
    for (const p of [1, 0]) {
      const { pts } = this.eqCurves[p];
      if (!pts.length) continue;
      ctx.strokeStyle = EQ_COLORS[p];
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      ctx.moveTo(px(0), py(pts[0].y));
      for (const pt of pts) ctx.lineTo(px(pt.x), py(pt.y));
      ctx.lineTo(px(1), py(pts[pts.length - 1].y));
      ctx.stroke();
    }
    ctx.lineWidth = 1;

    // legend
    [0, 1].forEach(p => {
      ctx.fillStyle = EQ_COLORS[p];
      ctx.beginPath(); ctx.arc(L + 14, T + 10 + p * 17, 4, 0, 7); ctx.fill();
      ctx.fillStyle = '#a8a8a8';
      ctx.fillText(this.posLabel(p), L + 24, T + 14 + p * 17);
    });

    // hovered/pinned hand combos as dots, on the viewed player's curve only
    const ref = this.hoverCell || this.selectedCell;
    if (ref) {
      const info = cellInfo(ref[0], ref[1]);
      const p = this.player;
      const { posByHand } = this.eqCurves[p];
      for (const [a, b] of cellCombos(info)) {
        const hi = this.handIdx[p].get(comboIndex(a, b));
        if (hi === undefined) continue;
        const pos = posByHand.get(hi);
        if (!pos) continue;
        ctx.fillStyle = '#f5c542';
        ctx.strokeStyle = '#1a1a1a';
        ctx.beginPath();
        ctx.arc(px(pos.x), py(pos.y), 4, 0, 7);
        ctx.fill(); ctx.stroke();
      }
    }

    // crosshair from chart hover: which hand sits at this percentile
    if (this.eqHoverX != null) {
      const xv = this.eqHoverX;
      ctx.strokeStyle = '#555';
      ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(px(xv), py(0)); ctx.lineTo(px(xv), py(1)); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#c8c8c8';
      ctx.fillText(`${Math.round(xv * 100)}`, px(xv) - 9, H - 7);
      const lines = [];
      for (const p of [0, 1]) {
        const { pts } = this.eqCurves[p];
        if (!pts.length) continue;
        let pt = pts[0];
        for (const q of pts) { if (q.x <= xv) pt = q; else break; }
        const h = this.view.players[p].hands[pt.i];
        lines.push({ p, text: `${this.posLabel(p)} ${h.combo} ${(pt.y * 100).toFixed(1)}%`, y: pt.y });
        ctx.fillStyle = EQ_COLORS[p];
        ctx.beginPath(); ctx.arc(px(pt.x), py(pt.y), 3.5, 0, 7); ctx.fill();
      }
      // label box
      const bw = 168;
      const bx = Math.min(px(xv) + 10, W - bw - 6);
      const by = T + 8;
      ctx.fillStyle = 'rgba(18,18,18,.92)';
      ctx.fillRect(bx, by, bw, 18 * lines.length + 10);
      lines.forEach((l, k) => {
        ctx.fillStyle = EQ_COLORS[l.p];
        ctx.fillText(l.text, bx + 8, by + 16 + 18 * k);
      });
    }
  }

  // ----- Blockers tab -----

  renderBlockersPanel(content) {
    const p = this.player;
    const isActor = this.view.node_type === 'action' && this.view.player === p;
    if (!isActor) {
      content.innerHTML = '<div class="dim" style="padding:10px 4px;font-size:11px">' +
        'Blocker effects are shown for the player to act — navigate to one of their decision nodes.</div>';
      return;
    }
    const hands = this.view.players[p].hands;
    const acts = this.view.actions;
    const na = acts.length;
    const colors = this.actionColors();
    const boardSet = new Set(this.view.board.map(cardFromString));

    // overall action mix + per-card mix among hands containing the card
    const overall = new Array(na).fill(0);
    let totalReach = 0;
    const perCard = new Map(); // card -> {reach, strat[]}
    hands.forEach(h => {
      if (!h.strategy || h.reach <= 0) return;
      totalReach += h.reach;
      h.strategy.forEach((s, k) => overall[k] += s * h.reach);
      for (const c of [h.c1, h.c2]) {
        let e = perCard.get(c);
        if (!e) { e = { reach: 0, strat: new Array(na).fill(0) }; perCard.set(c, e); }
        e.reach += h.reach;
        if (h.strategy) h.strategy.forEach((s, k) => e.strat[k] += s * h.reach);
      }
    });
    if (totalReach <= 1e-12) {
      content.innerHTML = '<div class="dim" style="padding:10px 4px;font-size:11px">no hands reach this node</div>';
      return;
    }
    const overallFreq = overall.map(x => x / totalReach);

    const rows = [];
    for (const [c, e] of perCard) {
      if (boardSet.has(c) || e.reach <= 1e-9) continue;
      const deltas = e.strat.map((x, k) => x / e.reach - overallFreq[k]);
      rows.push({ c, deltas });
    }
    if (this.blockerSort == null || (this.blockerSort.col !== 'card' && this.blockerSort.col >= na)) {
      // default: the most-played aggressive action (its deltas are the
      // meaningful blocker signal), falling back to the most-played action
      let k = 0, best = -1;
      for (let a = 0; a < na; a++) {
        const aggr = acts[a].kind === 'bet' || acts[a].kind === 'raise';
        const score = overallFreq[a] + (aggr ? 1 : 0);
        if (score > best) { best = score; k = a; }
      }
      this.blockerSort = { col: k, dir: -1 }; // desc: strongest positive first
    }
    const { col: sortCol, dir: sortDir } = this.blockerSort;
    const cmp = sortCol === 'card'
      ? (a, b) => a.c - b.c
      : (a, b) => a.deltas[sortCol] - b.deltas[sortCol];
    rows.sort((a, b) => (sortDir === -1 ? cmp(b, a) : cmp(a, b)));
    const maxAbs = rows.reduce((m, r) => Math.max(m, ...r.deltas.map(Math.abs)), 1e-9);

    const arrow = k =>
      k === sortCol ? (sortDir === -1 ? ' ▼' : ' ▲') : '';
    const header =
      `<div class="blocker-row head" style="grid-template-columns:52px repeat(${na},1fr)">` +
      `<span class="bk-head${sortCol === 'card' ? ' sorted' : ''}" data-sort="card" ` +
      `data-tip="Sort by card (rank, then suit). Click again to flip direction.">cards${arrow('card')}</span>` +
      acts.map((a, k) =>
        `<span class="bk-head${k === sortCol ? ' sorted' : ''}" data-sort="${k}" ` +
        `data-tip="Sort by this action's blocker shift — descending puts the strongest positive effect on top; click again for ascending (strongest negative)."><i style="background:${colors[k]}"></i>${a.label}${arrow(k)}</span>`
      ).join('') + `</div>`;

    const body = rows.map(r => {
      const cardHtml =
        `<span class="bk-card"><b>${RANKS[rank(r.c)]}</b><span class="suit-${SUITS[suit(r.c)]}">${SUIT_GLYPH[SUITS[suit(r.c)]]}</span></span>`;
      const cells = r.deltas.map(d => {
        const t = Math.min(1, Math.abs(d) / maxAbs);
        const bg = d >= 0
          ? `hsl(${70 + 50 * t} 50% ${40 - 6 * t}%)`
          : `hsl(${35 - 27 * t} 65% ${46 - 5 * t}%)`;
        return `<span class="bk-cell" style="background:${bg}">${d >= 0 ? '+' : '−'} ${(Math.abs(d) * 100).toFixed(2)}%</span>`;
      }).join('');
      return `<div class="blocker-row" style="grid-template-columns:52px repeat(${na},1fr)">${cardHtml}${cells}</div>`;
    }).join('');

    content.innerHTML =
      `<div class="dim" style="font-size:11px;margin-bottom:8px">How holding each card shifts ` +
      `${this.posLabel(p)}'s strategy vs the range average — the essence of blocker selection.</div>` +
      header + `<div class="blocker-body">${body}</div>`;

    content.querySelectorAll('.bk-head').forEach(h =>
      h.addEventListener('click', () => {
        const col = h.dataset.sort === 'card' ? 'card' : +h.dataset.sort;
        if (this.blockerSort && this.blockerSort.col === col) {
          this.blockerSort.dir *= -1; // same column: flip direction
        } else {
          this.blockerSort = { col, dir: -1 };
        }
        this.renderBlockersPanel(content);
      }));
  }

  summaryHtml(present, isActor, colors, acts) {
    let reach = 0, weight = 0, ev = 0, evW = 0, eq = 0, eqW = 0;
    const freqs = acts.map(() => 0);
    const aev = acts.map(() => ({ n: 0, d: 0 }));
    for (const [h] of present) {
      reach += h.reach; weight += h.weight;
      if (h.ev != null) { ev += h.ev * h.reach; evW += h.reach; }
      if (h.eq != null) { eq += h.eq * h.reach; eqW += h.reach; }
      if (isActor && h.strategy) {
        h.strategy.forEach((s, k) => {
          freqs[k] += s * h.reach;
          if (h.evs && h.evs[k] != null) { aev[k].n += h.evs[k] * h.reach * s; aev[k].d += h.reach * s; }
        });
      }
    }
    const stats = `<div class="summary-stats">
      <div class="stat"><label>combos</label><div>${present.length}</div></div>
      <div class="stat"><label>weight</label><div>${weight.toFixed(1)}</div></div>
      <div class="stat"><label>avg EQ</label><div>${eqW > 1e-9 ? (eq / eqW * 100).toFixed(1) + '%' : '—'}</div></div>
      <div class="stat"><label>avg EV</label><div>${evW > 1e-9 ? fmt(ev / evW) : '—'}</div></div>
    </div>`;
    if (!isActor || reach <= 1e-9) return stats;
    const bar = `<div class="summary-bar">` + freqs.map((f, k) =>
      `<div style="width:${(f / reach * 100).toFixed(1)}%;background:${colors[k]}"></div>`).join('') + `</div>`;
    const rows = acts.map((a, k) => {
      const f = reach > 1e-9 ? freqs[k] / reach : 0;
      const e = aev[k].d > 1e-9 ? fmt(aev[k].n / aev[k].d) : '—';
      return `<div class="summary-row"><span class="swatch" style="background:${colors[k]}"></span>` +
        `<span class="lab">${a.label}</span><span class="num"><b>${(f * 100).toFixed(1)}%</b> · EV ${e}</span></div>`;
    }).join('');
    return stats + bar + rows;
  }

  syncSegs() {
    this.els.segPlayer.querySelectorAll('button').forEach(b => {
      const p = +b.dataset.v;
      b.classList.toggle('active', p === this.player);
      b.textContent = this.posLabel(p); // BB/BTN… from preflop setup, else OOP/IP
    });
    this.els.segMode.querySelectorAll('button').forEach(b =>
      b.classList.toggle('active', b.dataset.v === this.mode));
  }
}

function fmt(x) {
  if (x == null || Number.isNaN(x)) return '—';
  if (Math.abs(x) < 0.005) x = 0; // avoid "-0.00"
  const a = Math.abs(x);
  if (a >= 100) return x.toFixed(0);
  if (a >= 10) return x.toFixed(1);
  return x.toFixed(2);
}

function computeActionColors(actions, pot) {
  return actions.map(a => {
    if (a.kind === 'bet' || a.kind === 'raise') {
      return betShade(a.amount, pot);
    }
    return ACTION_COLORS[a.kind] || '#888';
  });
}

export function cardChip(cs, cls = 'bcard') {
  const d = document.createElement('div');
  const r = cs[0], s = cs[1];
  d.className = `${cls} cbg-${s}`;
  d.innerHTML = `<span class="rank">${r}</span><span class="pip">${SUIT_GLYPH[s]}</span>`;
  return d;
}

export function facedownChip(cls = 'bcard') {
  const d = document.createElement('div');
  d.className = `${cls} facedown`;
  return d;
}
