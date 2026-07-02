// Preflop "study" context for FREEPIO.
//
// The solver is heads-up postflop only. This module lets you configure a game
// and a preflop ACTION LINE that must end with exactly two players seeing the
// flop, then DERIVES the postflop spot (pot, effective stack, which two
// positions are OOP/IP) for the existing engine. It does NOT solve preflop —
// the two continuing ranges are supplied by the user (chart import / editing).
//
// Everything is in big blinds (bb); the engine only cares about pot:stack
// ratios, so bb units feed straight into /api/spot.

// 6-max. Preflop action order (UTG acts first); postflop order (SB acts first,
// BTN last) gives the OOP/IP assignment for the two flop players.
export const POSITIONS = ['UTG', 'HJ', 'CO', 'BTN', 'SB', 'BB'];
const ACT_ORDER = ['UTG', 'HJ', 'CO', 'BTN', 'SB', 'BB'];        // preflop turn order
const POSTFLOP_RANK = { SB: 0, BB: 1, UTG: 2, HJ: 3, CO: 4, BTN: 5 };
const SB = 0.5, BB = 1.0;

export function freshState(stackBb = 100) {
  return { stackBb, ante: 0, line: [] }; // line: [{pos, type:'fold'|'call'|'raise'|'check', toBb?}]
}

// Replay the line to current investments / fold state / amount-to-call.
function replay(state) {
  const invested = {}, acted = new Set(), folded = new Set();
  for (const p of POSITIONS) invested[p] = state.ante || 0;
  invested.SB += SB;
  invested.BB += BB;
  let toCall = BB;
  // Players who still owe an action. BB carries the "option" when limped.
  let needs = new Set(POSITIONS);
  for (const a of state.line) {
    acted.add(a.pos);
    needs.delete(a.pos);
    if (a.type === 'fold') {
      folded.add(a.pos);
    } else if (a.type === 'call' || a.type === 'check') {
      invested[a.pos] = Math.max(invested[a.pos], toCall);
    } else if (a.type === 'raise') {
      invested[a.pos] = a.toBb;
      toCall = a.toBb;
      // a raise re-opens action for every other live player
      for (const p of POSITIONS) if (!folded.has(p) && p !== a.pos) needs.add(p);
    }
  }
  return { invested, acted, folded, toCall, needs };
}

// Whose turn is it (null = action closed for the street)?
export function nextActor(state) {
  const { folded, needs } = replay(state);
  if (!state.line.length) return 'UTG';
  const last = state.line[state.line.length - 1].pos;
  const start = ACT_ORDER.indexOf(last);
  for (let k = 1; k <= POSITIONS.length; k++) {
    const p = ACT_ORDER[(start + k) % POSITIONS.length];
    if (!folded.has(p) && needs.has(p)) return p;
  }
  return null;
}

// Legal actions for the player to act, as {type, label, toBb?}.
export function legalActions(state) {
  const pos = nextActor(state);
  if (!pos) return [];
  const { invested, toCall, folded } = replay(state);
  const live = POSITIONS.filter(p => !folded.has(p)).length;
  const out = [];
  // Fold is illegal when you can check for free (you're the BB with no raise).
  const canCheck = invested[pos] >= toCall - 1e-9;
  if (!canCheck) out.push({ type: 'fold', label: 'Fold' });
  if (canCheck) out.push({ type: 'check', label: 'Check' });
  else out.push({ type: 'call', label: `Call ${fmtBb(toCall)}` });
  // Raise/3-bet/4-bet: only meaningful while >1 player can still be in.
  if (live > 1) {
    const nRaises = state.line.filter(a => a.type === 'raise').length;
    const isOpen = nRaises === 0;
    const suggest = isOpen
      ? roundBb(2.5)                                // default open 2.5bb
      : roundBb(toCall * 3);                        // default re-raise 3x
    const label = isOpen ? 'Raise' : `${nRaises + 2}-bet`;
    out.push({ type: 'raise', label, toBb: suggest });
  }
  return out;
}

// Derive the postflop spot, or an explanation of why the line isn't ready.
export function derive(state) {
  const { invested, folded, toCall } = replay(state);
  const closed = nextActor(state) === null;
  // Flop players: not folded and matched the final bet.
  const flop = POSITIONS.filter(p => !folded.has(p) && Math.abs(invested[p] - toCall) < 1e-9);
  if (!closed) return { ready: false, reason: 'preflop action is still open' };
  if (flop.length < 2) return { ready: false, reason: 'everyone folded — need two players to the flop' };
  if (flop.length > 2) return { ready: false, reason: `${flop.length} players reach the flop; this engine is heads-up (exactly 2)` };
  const [a, b] = flop;
  const oop = POSTFLOP_RANK[a] < POSTFLOP_RANK[b] ? a : b;
  const ip = oop === a ? b : a;
  const pot = POSITIONS.reduce((s, p) => s + invested[p], 0);
  // both flop players invested toCall, plus their ante
  const effStack = state.stackBb - toCall - (state.ante || 0);
  return {
    ready: true, oop, ip, potBb: round1(pot), effStackBb: round1(effStack),
    investedBb: invested, line: state.line.slice(),
  };
}

// A short, human description of the line for the ribbon / summary.
export function lineSummary(state) {
  if (!state.line.length) return '(no preflop action)';
  return state.line.map(a =>
    a.type === 'raise' ? `${a.pos} ${fmtBb(a.toBb)}`
      : a.type === 'call' ? `${a.pos} call`
      : a.type === 'check' ? `${a.pos} check`
      : `${a.pos} fold`).join(' · ');
}

// Ribbon segments for the preflop street: one per non-fold continuing action of
// the two flop players (folds are dead money, not shown as nodes).
export function preflopSegments(state) {
  const d = derive(state);
  if (!d.ready) return [];
  return state.line
    .filter(a => a.type !== 'fold')
    .map(a => ({
      pos: a.pos,
      label: a.type === 'raise' ? fmtBb(a.toBb) : a.type === 'call' ? 'call' : 'check',
      potBb: null,
    }));
}

export const PRESETS = [
  { name: 'BTN open, BB call (SRP)', stackBb: 100, line: [
    { pos: 'UTG', type: 'fold' }, { pos: 'HJ', type: 'fold' }, { pos: 'CO', type: 'fold' },
    { pos: 'BTN', type: 'raise', toBb: 2.5 }, { pos: 'SB', type: 'fold' }, { pos: 'BB', type: 'call' } ] },
  { name: 'CO open, BTN 3-bet, CO call (3BP)', stackBb: 100, line: [
    { pos: 'UTG', type: 'fold' }, { pos: 'HJ', type: 'fold' }, { pos: 'CO', type: 'raise', toBb: 2.5 },
    { pos: 'BTN', type: 'raise', toBb: 7.5 }, { pos: 'SB', type: 'fold' }, { pos: 'BB', type: 'fold' },
    { pos: 'CO', type: 'call' } ] },
  { name: 'SB limp, BB check (limped)', stackBb: 100, line: [
    { pos: 'UTG', type: 'fold' }, { pos: 'HJ', type: 'fold' }, { pos: 'CO', type: 'fold' },
    { pos: 'BTN', type: 'fold' }, { pos: 'SB', type: 'call' }, { pos: 'BB', type: 'check' } ] },
  { name: 'BTN open, BB 3-bet, BTN call (3BP)', stackBb: 100, line: [
    { pos: 'UTG', type: 'fold' }, { pos: 'HJ', type: 'fold' }, { pos: 'CO', type: 'fold' },
    { pos: 'BTN', type: 'raise', toBb: 2.5 }, { pos: 'SB', type: 'fold' }, { pos: 'BB', type: 'raise', toBb: 10 },
    { pos: 'BTN', type: 'call' } ] },
];

function roundBb(x) { return Math.round(x * 2) / 2; }   // nearest 0.5bb
function round1(x) { return Math.round(x * 10) / 10; }
function fmtBb(x) { return (Math.round(x * 10) / 10) + ''; }
