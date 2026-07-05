// Hand-category classification for the Filters tab:
// made-hand tiers, draw types, and equity buckets.

import { rank, suit } from './cards.js';

export const MADE_LABELS = {
  sf: 'Straight flush', quads: 'Quads', boat: 'Full house', flush: 'Flush',
  straight: 'Straight', set: 'Set', trips: 'Trips', two_pair: 'Two pair',
  overpair: 'Overpair', top_pair: 'Top pair', underpair: 'Underpair',
  second_pair: 'Second pair', third_pair: 'Third pair', weak_pair: 'Weak pair',
  ace_high: 'Ace high', king_high: 'King high', no_made: 'No made hand',
};
export const MADE_ORDER = Object.keys(MADE_LABELS);

export const DRAW_LABELS = {
  combo: 'Combo draw', nut_fd: 'Flush draw nuts', fd: 'Flush draw',
  oesd: 'OESD', gutshot: 'Gutshot', bdfd2: 'BDFD 2 cards', bdfd1: 'BDFD 1 card',
  no_draw: 'No draw',
};
export const DRAW_ORDER = Object.keys(DRAW_LABELS);

export const EQS_LABELS = {
  eq_best: 'Best hands', eq_good: 'Good hands', eq_weak: 'Weak hands', eq_trash: 'Trash hands',
};
export const EQA_LABELS = {
  eqa_90: 'Equity 90-100', eqa_80: 'Equity 80-90', eqa_70: 'Equity 70-80',
  eqa_60: 'Equity 60-70', eqa_50: 'Equity 50-60', eqa_25: 'Equity 25-50',
  eqa_0: 'Equity 0-25',
};

function hasStraight(rankMask) {
  const m = (rankMask << 1) | ((rankMask >> 12) & 1); // ace low
  let run = 0;
  for (let i = 0; i < 14; i++) {
    if (m & (1 << i)) { if (++run >= 5) return true; } else run = 0;
  }
  return false;
}

/** Classify one hand vs a board (array of card ints). Returns {made, draw}. */
export function classify(board, c1, c2, eq) {
  const hr = [rank(c1), rank(c2)];
  const hs = [suit(c1), suit(c2)];
  const br = board.map(rank);
  const pocket = hr[0] === hr[1];
  const all = [...board, c1, c2];

  const rankCountAll = new Array(13).fill(0);
  const rankCountBoard = new Array(13).fill(0);
  let rankMaskAll = 0, rankMaskBoard = 0;
  for (const r of br) { rankCountBoard[r]++; rankMaskBoard |= 1 << r; }
  for (const c of all) { rankCountAll[rank(c)]++; rankMaskAll |= 1 << rank(c); }

  const suitCountAll = [0, 0, 0, 0];
  const suitCountBoard = [0, 0, 0, 0];
  for (const c of board) suitCountBoard[suit(c)]++;
  for (const c of all) suitCountAll[suit(c)]++;

  const boardTop = Math.max(...br);
  // distinct board ranks, descending
  const boardRanks = [...new Set(br)].sort((a, b) => b - a);

  let made = null;

  // flush / straight flush
  let flushSuit = -1;
  for (let s = 0; s < 4; s++) if (suitCountAll[s] >= 5) flushSuit = s;
  if (flushSuit >= 0) {
    let sfMask = 0;
    for (const c of all) if (suit(c) === flushSuit) sfMask |= 1 << rank(c);
    if (hasStraight(sfMask)) made = 'sf';
  }
  const quadRank = rankCountAll.findIndex(n => n === 4);
  const tripRanks = [];
  const pairRanks = [];
  for (let r = 12; r >= 0; r--) {
    if (rankCountAll[r] === 3) tripRanks.push(r);
    if (rankCountAll[r] === 2) pairRanks.push(r);
  }
  if (!made && quadRank >= 0) made = 'quads';
  if (!made && tripRanks.length && (tripRanks.length > 1 || pairRanks.length)) made = 'boat';
  if (!made && flushSuit >= 0) made = 'flush';
  if (!made && hasStraight(rankMaskAll)) made = 'straight';
  if (!made && tripRanks.length) {
    const r = tripRanks[0];
    if (rankCountBoard[r] < 3) {
      // hole card participates: pocket pair hitting = set, board pair + hole = trips
      made = pocket && hr[0] === r ? 'set' : 'trips';
    }
    // board trips without hole involvement falls through to pair/high-card tiers
  }
  if (!made) {
    // pairs where a hole card participates
    const holePairs = pairRanks.filter(r => hr.includes(r) && rankCountBoard[r] < 2);
    if (pairRanks.length >= 2 && holePairs.length >= 1) made = 'two_pair';
    else if (holePairs.length === 1 || (pocket && rankCountBoard[hr[0]] === 0)) {
      if (pocket) {
        made = hr[0] > boardTop ? 'overpair' : 'underpair';
      } else {
        const r = holePairs[0];
        const pos = boardRanks.indexOf(r);
        made = pos === 0 ? 'top_pair' : pos === 1 ? 'second_pair' : pos === 2 ? 'third_pair' : 'weak_pair';
      }
    }
  }
  if (!made) {
    const hiHole = Math.max(...hr);
    made = hiHole === 12 ? 'ace_high' : hiHole === 11 ? 'king_high' : 'no_made';
  }

  // ---- draws (only before the river, and not for monsters) ----
  let draw = 'no_draw';
  if (board.length < 5) {
    const strongMade = ['sf', 'quads', 'boat', 'flush', 'straight'].includes(made);
    // flush draw: exactly 4 of a suit using >= 1 hole card
    let fdSuit = -1;
    for (let s = 0; s < 4; s++) {
      const holeOf = (hs[0] === s ? 1 : 0) + (hs[1] === s ? 1 : 0);
      if (suitCountAll[s] === 4 && holeOf >= 1) fdSuit = s;
    }
    // straight outs: ranks that complete a straight not already on board alone
    let outs = 0;
    if (!strongMade && made !== 'straight') {
      for (let r = 0; r < 13; r++) {
        const withR = rankMaskAll | (1 << r);
        const boardWithR = rankMaskBoard | (1 << r);
        if (hasStraight(withR) && !hasStraight(boardWithR)) outs++;
      }
    }
    const nutFd = (() => {
      if (fdSuit < 0) return false;
      // highest rank of the suit not on the board
      for (let r = 12; r >= 0; r--) {
        const onBoard = board.some(c => suit(c) === fdSuit && rank(c) === r);
        if (onBoard) continue;
        return (hs[0] === fdSuit && hr[0] === r) || (hs[1] === fdSuit && hr[1] === r);
      }
      return false;
    })();

    if (strongMade) draw = 'no_draw';
    else if (fdSuit >= 0 && outs >= 1) draw = 'combo';
    else if (nutFd) draw = 'nut_fd';
    else if (fdSuit >= 0) draw = 'fd';
    else if (outs >= 2) draw = 'oesd';
    else if (outs === 1) draw = 'gutshot';
    else if (board.length === 3) {
      // backdoor flush draws
      for (let s = 0; s < 4; s++) {
        const holeOf = (hs[0] === s ? 1 : 0) + (hs[1] === s ? 1 : 0);
        if (suitCountAll[s] === 3 && holeOf === 2) { draw = 'bdfd2'; break; }
        if (suitCountAll[s] === 3 && holeOf === 1) draw = 'bdfd1';
      }
    }
  }

  // ---- equity buckets ----
  let eqs = null, eqa = null;
  if (eq != null) {
    eqs = eq >= 0.75 ? 'eq_best' : eq >= 0.5 ? 'eq_good' : eq >= 0.25 ? 'eq_weak' : 'eq_trash';
    eqa = eq >= 0.9 ? 'eqa_90' : eq >= 0.8 ? 'eqa_80' : eq >= 0.7 ? 'eqa_70'
        : eq >= 0.6 ? 'eqa_60' : eq >= 0.5 ? 'eqa_50' : eq >= 0.25 ? 'eqa_25' : 'eqa_0';
  }

  return { made, draw, eqs, eqa };
}

/** Suited/offsuit + suit-content predicates for the suit filter row. */
export function suitTags(c1, c2) {
  const tags = new Set();
  if (suit(c1) === suit(c2)) {
    tags.add(`s${suit(c1)}`);
  } else {
    tags.add(`o${suit(c1)}`);
    tags.add(`o${suit(c2)}`);
  }
  return tags;
}
