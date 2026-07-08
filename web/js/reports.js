// FLOP REPORTS — solve the current SETUP spot across a canonical flop
// subset and study aggregate strategy/EV/EQ/EQR by texture, GTO-style.
// Chart = one thin stacked frequency bar per flop (the app's semantic
// action colors, aggressive→passive fixed order), hover tooltips, click
// to inspect + open in Browse. Table + legend ship alongside (identity
// is never color-alone).

import { api } from './api.js';

const RANKS = '23456789TJQKA';

export function initReports({ els, toast, currentSpot, villains, openInBrowse }) {
  const S = {
    report: null,      // loaded report json
    sort: { key: 'rank', dir: -1 },
    tex: 'all',
    node: 'root',      // 'root' (OOP first decision) | 'vs_check' (IP reply)
    selected: null,    // board string
    polling: null,
  };

  // ---------------------------------------------------------- helpers ----

  const cardsOf = b => [b.slice(0, 2), b.slice(2, 4), b.slice(4, 6)];
  function texOf(board) {
    const cs = cardsOf(board);
    const rs = cs.map(c => RANKS.indexOf(c[0])).sort((a, b) => b - a);
    const suits = new Set(cs.map(c => c[1]));
    const paired = new Set(rs).size < 3;
    const span = rs[0] - rs[2];
    return {
      all: true,
      rainbow: suits.size === 3,
      twotone: suits.size === 2,
      mono: suits.size === 1,
      paired,
      connected: !paired && span <= 4,
      acehigh: rs[0] === 12,
      broadway: rs[0] === 11 || rs[0] === 10,
      mid: rs[0] >= 7 && rs[0] <= 9,
      low: rs[0] <= 6,
    };
  }
  const TEX = [
    ['all', 'ALL'], ['rainbow', 'RAINBOW'], ['twotone', 'TWO-TONE'],
    ['mono', 'MONO'], ['paired', 'PAIRED'], ['connected', 'CONNECTED'],
    ['acehigh', 'A-HIGH'], ['broadway', 'K/Q-HIGH'], ['mid', 'MID'], ['low', 'LOW'],
  ];

  // action colors: the app's semantic palette (fold blue, check/call green,
  // bets by size in reds, jam purple) — same mapping as every other view
  function actionColor(kind, label, idx, n) {
    if (kind === 'fold') return '#4a78c8';
    if (kind === 'check' || kind === 'call') return '#5ca75f';
    if (/All-in/i.test(label)) return '#7d3ca3';
    const reds = ['#e8484c', '#c73e55', '#a4335f'];
    return reds[Math.min(idx, reds.length - 1)];
  }
  function stratColors(strat) {
    // strat.kinds/labels; index bets by their order among aggressive acts
    let bi = 0;
    return strat.kinds.map((k, i) => {
      const c = actionColor(k, strat.actions[i], bi, strat.kinds.length);
      if (k === 'bet' || k === 'raise') bi++;
      return c;
    });
  }
  const stratOf = row => (S.node === 'root' ? row.root : row.vs_check) || null;
  const aggrPct = row => {
    const st = stratOf(row);
    if (!st) return 0;
    return st.freqs.reduce((s, f, i) =>
      s + ((st.kinds[i] === 'bet' || st.kinds[i] === 'raise') ? f : 0), 0);
  };
  const metric = (row, key) => {
    const P = row.players;
    switch (key) {
      case 'bet': return aggrPct(row);
      case 'ev0': return P[0].ev; case 'ev1': return P[1].ev;
      case 'eq0': return P[0].eq; case 'eq1': return P[1].eq;
      case 'eqr0': return P[0].eqr; case 'eqr1': return P[1].eqr;
      default: {
        const rs = cardsOf(row.board).map(c => RANKS.indexOf(c[0]));
        return rs[0] * 169 + rs[1] * 13 + rs[2];
      }
    }
  };

  function visibleRows() {
    if (!S.report) return [];
    const rows = S.report.flops.filter(r => texOf(r.board)[S.tex]);
    const { key, dir } = S.sort;
    return rows.slice().sort((a, b) => dir * (metric(b, key) - metric(a, key)));
  }

  // ---------------------------------------------------------- library ----

  async function refreshLibrary() {
    let list = [];
    try { list = await api.reportsList(); } catch { return; }
    els.library.innerHTML = '';
    if (!list.length) {
      els.library.innerHTML =
        '<div class="dim" style="font-size:11px;padding:6px 2px">no reports yet — configure a spot in SETUP and run one</div>';
      return;
    }
    for (const r of list) {
      const row = document.createElement('button');
      row.className = 'report-item';
      const when = r.created ? new Date(r.created * 1000).toISOString().slice(0, 10) : '';
      row.innerHTML = `<b>${r.name}</b><span class="dim">${r.n_flops} flops` +
        `${r.villain ? ' · vs ' + r.villain : ''}${r.complete ? '' : ' · PARTIAL'} · ${when}</span>`;
      row.addEventListener('click', () => loadReport(r.name));
      els.library.appendChild(row);
    }
  }

  async function loadReport(name) {
    try {
      S.report = await api.reportsGet(name);
      S.selected = null;
      render();
    } catch (e) { toast(e.message, true); }
  }

  // ------------------------------------------------------------- run ----

  els.run.addEventListener('click', async () => {
    const spot = currentSpot();
    if (!spot) return toast('configure a spot in SETUP first (ranges + sizes)', true);
    const name = els.name.value.trim() || `report ${new Date().toISOString().slice(0, 16).replace('T', ' ')}`;
    const body = {
      name, spot,
      flops: +els.flops.value,
      max_iterations: 600,
      target: 0.35,
    };
    const v = villains();
    if (els.vsVillain.checked && v) body.villain = v;
    try {
      await api.reportsRun(body);
      toast(`report "${name}" running — ${body.flops} flops${body.villain ? ' vs ' + body.villain.name : ''}`);
      pollStatus();
    } catch (e) { toast(e.message, true); }
  });
  els.stop.addEventListener('click', () => api.reportsStop().catch(() => {}));

  function pollStatus() {
    if (S.polling) clearInterval(S.polling);
    S.polling = setInterval(async () => {
      let st;
      try { st = await api.reportsStatus(); } catch { return; }
      els.stop.classList.toggle('hidden', !st.running);
      els.run.classList.toggle('hidden', st.running);
      if (st.running) {
        els.progress.textContent =
          `${st.name}: ${st.done}/${st.total} · ${st.board} · ${(st.seconds / 60).toFixed(1)} min`;
      } else {
        if (els.progress.textContent) {
          els.progress.textContent = st.error ? `failed: ${st.error}` : '';
          if (st.error) toast(st.error, true);
          else if (st.name) { toast(`report "${st.name}" done`); refreshLibrary(); loadReport(st.name); }
        }
        clearInterval(S.polling);
        S.polling = null;
      }
    }, 2000);
  }

  // ----------------------------------------------------------- viewer ----

  function render() {
    const rep = S.report;
    els.viewer.classList.toggle('hidden', !rep);
    if (!rep) return;
    const v = rep.villain ? ` · villain: ${rep.villain.name}` : '';
    els.title.textContent = `${rep.name} — ${rep.flops.length} flops${v}`;
    els.subtitle.textContent =
      `pot ${rep.spot.starting_pot} · stack ${rep.spot.effective_stack} · rake ${rep.spot.rake_pct}%` +
      ` · target ${rep.target_pct}% pot${rep.complete ? '' : ' · PARTIAL RUN'}`;

    // controls (idempotent rebuild)
    els.controls.innerHTML =
      `<div class="seg" id="rep-node">` +
      `<button data-n="root" class="${S.node === 'root' ? 'active' : ''}" data-tip="The first decision on the flop (OOP acting into the pot).">OOP ROOT</button>` +
      `<button data-n="vs_check" class="${S.node === 'vs_check' ? 'active' : ''}" data-tip="IP's reply after OOP checks — the c-bet view.">IP VS CHECK</button></div>` +
      `<select id="rep-sort" data-tip="Order the flop strip.">` +
      ['rank|board', 'bet|bet %', 'ev0|OOP EV', 'ev1|IP EV', 'eq0|OOP EQ', 'eq1|IP EQ', 'eqr0|OOP EQR', 'eqr1|IP EQR']
        .map(o => { const [k, l] = o.split('|'); return `<option value="${k}" ${S.sort.key === k ? 'selected' : ''}>${l}</option>`; }).join('') +
      `</select>` +
      `<div class="seg" id="rep-tex">` +
      TEX.map(([k, l]) => `<button data-t="${k}" class="${S.tex === k ? 'active' : ''}">${l}</button>`).join('') +
      `</div>`;
    els.controls.querySelectorAll('#rep-node button').forEach(b =>
      b.addEventListener('click', () => { S.node = b.dataset.n; render(); }));
    els.controls.querySelector('#rep-sort').addEventListener('change', e => {
      S.sort = { key: e.target.value, dir: -1 }; render();
    });
    els.controls.querySelectorAll('#rep-tex button').forEach(b =>
      b.addEventListener('click', () => { S.tex = b.dataset.t; render(); }));

    const rows = visibleRows();
    drawStrip(rows);
    renderAggregate(rows);
    renderTable(rows);
    renderDetail();
    renderLegend(rows);
  }

  function renderLegend(rows) {
    els.legend.innerHTML = '';
    const st = rows.map(stratOf).find(x => x);
    if (!st) return;
    const colors = stratColors(st);
    st.actions.forEach((a, i) => {
      els.legend.innerHTML += `<span class="key"><i style="background:${colors[i]}"></i>${a}</span>`;
    });
  }

  function renderAggregate(rows) {
    if (!rows.length) { els.aggregate.innerHTML = ''; return; }
    const st0 = rows.map(stratOf).find(x => x);
    if (!st0) { els.aggregate.innerHTML = ''; return; }
    const na = st0.freqs.length;
    const sums = new Array(na).fill(0);
    let wtot = 0;
    const m = { ev0: 0, ev1: 0, eq0: 0, eqr0: 0 };
    for (const r of rows) {
      const st = stratOf(r);
      const w = r.weight || 1;
      wtot += w;
      if (st) for (let a = 0; a < na; a++) sums[a] += (st.freqs[a] || 0) * w;
      m.ev0 += r.players[0].ev * w; m.ev1 += r.players[1].ev * w;
      m.eq0 += r.players[0].eq * w; m.eqr0 += r.players[0].eqr * w;
    }
    const colors = stratColors(st0);
    const bar = sums.map((s, a) =>
      `<div style="width:${(100 * s / wtot).toFixed(1)}%;background:${colors[a]}" data-tip="${st0.actions[a]}: ${(100 * s / wtot).toFixed(1)}% weighted over ${rows.length} flops"></div>`).join('');
    els.aggregate.innerHTML =
      `<span class="cname" data-tip="Iso-weighted average over the ${rows.length} flops shown.">avg·${rows.length}</span>` +
      `<span class="cbar">${bar}</span>` +
      `<span class="cnum">${(m.ev0 / wtot).toFixed(2)}</span><span class="cnum">${(m.ev1 / wtot).toFixed(2)}</span>` +
      `<span class="cnum">${(100 * m.eq0 / wtot).toFixed(1)}</span><span class="cnum">${(m.eqr0 / wtot).toFixed(2)}</span>`;
  }

  function drawStrip(rows) {
    const cv = els.canvas;
    const W = cv.clientWidth || 1100;
    const H = 190;
    const dpr = window.devicePixelRatio || 1;
    cv.width = W * dpr; cv.height = H * dpr;
    const ctx = cv.getContext('2d');
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H);
    if (!rows.length) return;
    const bw = Math.max(2, Math.floor(W / rows.length) - 1);
    const step = W / rows.length;
    S.hitmap = [];
    rows.forEach((r, i) => {
      const st = stratOf(r);
      const x = Math.floor(i * step);
      S.hitmap.push({ x0: x, x1: x + step, row: r });
      if (!st) return;
      const colors = stratColors(st);
      // draw passive at the bottom, aggressive stacked on top (fixed order)
      let y = H - 14;
      for (let a = st.freqs.length - 1; a >= 0; a--) {
        const hgt = st.freqs[a] * (H - 18);
        ctx.fillStyle = colors[a];
        ctx.fillRect(x, y - hgt, bw, hgt);
        y -= hgt;
      }
      if (r.board === S.selected) {
        ctx.strokeStyle = '#e6e6e6';
        ctx.strokeRect(x - 0.5, 1.5, bw + 1, H - 16);
      }
    });
    ctx.fillStyle = '#5a5a5a';
    ctx.font = '9px IBM Plex Mono, monospace';
    ctx.fillText(`${rows.length} flops · sorted by ${S.sort.key} · bars = ${S.node === 'root' ? 'OOP root strategy' : 'IP vs check'}`, 4, H - 3);
  }

  function rowAt(ev) {
    const rect = els.canvas.getBoundingClientRect();
    const x = ev.clientX - rect.left;
    return (S.hitmap || []).find(h => x >= h.x0 && x < h.x1)?.row || null;
  }
  els.canvas.addEventListener('mousemove', ev => {
    const r = rowAt(ev);
    if (!r) { els.canvas.removeAttribute('data-tip'); return; }
    const st = stratOf(r);
    const parts = st ? st.actions.map((a, i) => `${a} ${(100 * st.freqs[i]).toFixed(0)}%`).join(' · ') : '';
    els.canvas.dataset.tip =
      `${fmtBoard(r.board)} — ${parts} · OOP EV ${r.players[0].ev.toFixed(2)} · EQ ${(100 * r.players[0].eq).toFixed(1)}% · EQR ${r.players[0].eqr.toFixed(2)}`;
  });
  els.canvas.addEventListener('click', ev => {
    const r = rowAt(ev);
    if (r) { S.selected = r.board; render(); }
  });

  const SUIT_GLYPH = { c: '♣', d: '♦', h: '♥', s: '♠' };
  const fmtBoard = b => cardsOf(b).map(c => c[0] + SUIT_GLYPH[c[1]]).join('');

  function renderDetail() {
    const r = S.report && S.selected
      ? S.report.flops.find(x => x.board === S.selected) : null;
    els.detail.classList.toggle('hidden', !r);
    if (!r) return;
    const st = stratOf(r);
    els.detail.innerHTML =
      `<b class="mono">${fmtBoard(r.board)}</b> ` +
      `<span class="dim mono" style="font-size:11px">exploit ${r.exploit_pct.toFixed(2)}% · ` +
      (st ? st.actions.map((a, i) => `${a} ${(100 * st.freqs[i]).toFixed(1)}%`).join(' · ') : '') +
      ` · OOP ev ${r.players[0].ev.toFixed(2)} eq ${(100 * r.players[0].eq).toFixed(1)} eqr ${r.players[0].eqr.toFixed(2)}` +
      ` · IP ev ${r.players[1].ev.toFixed(2)} eq ${(100 * r.players[1].eq).toFixed(1)} eqr ${r.players[1].eqr.toFixed(2)}</span> ` +
      `<button class="btn ghost xs" id="rep-open" data-tip="Load this exact spot + board into SETUP, build and solve it, then study it in Browse.">OPEN IN BROWSE</button>`;
    els.detail.querySelector('#rep-open').addEventListener('click', () =>
      openInBrowse(S.report.spot, r.board));
  }

  function renderTable(rows) {
    const el = els.table;
    el.innerHTML = '';
    const head = document.createElement('div');
    head.className = 'combo-row head';
    head.innerHTML = `<span class="cname">flop</span><span class="cbar" style="background:none">strategy</span>` +
      `<span class="cnum">OOP EV</span><span class="cnum">IP EV</span><span class="cnum">OOP EQ</span><span class="cnum">OOP EQR</span>`;
    el.appendChild(head);
    for (const r of rows.slice(0, 200)) {
      const st = stratOf(r);
      const colors = st ? stratColors(st) : [];
      const bar = st ? st.freqs.map((f, a) =>
        `<div style="width:${(f * 100).toFixed(1)}%;background:${colors[a]}" data-tip="${st.actions[a]}: ${(f * 100).toFixed(1)}%"></div>`).join('') : '';
      const row = document.createElement('div');
      row.className = 'combo-row' + (r.board === S.selected ? ' sel' : '');
      row.innerHTML = `<span class="cname mono">${fmtBoard(r.board)}</span><span class="cbar">${bar}</span>` +
        `<span class="cnum">${r.players[0].ev.toFixed(2)}</span><span class="cnum">${r.players[1].ev.toFixed(2)}</span>` +
        `<span class="cnum">${(100 * r.players[0].eq).toFixed(1)}</span><span class="cnum">${r.players[0].eqr.toFixed(2)}</span>`;
      row.addEventListener('click', () => { S.selected = r.board; render(); });
      el.appendChild(row);
    }
  }

  refreshLibrary();
  pollStatus();
  return { refreshLibrary };
}
