//! Card representation and parsing.
//!
//! A card is a `u8` in `0..52`, encoded as `rank * 4 + suit`.
//! Ranks: 0 = deuce .. 12 = ace. Suits: 0 = clubs, 1 = diamonds, 2 = hearts, 3 = spades.

pub type Card = u8;

pub const NUM_CARDS: usize = 52;
/// Number of distinct two-card combos: C(52, 2).
pub const NUM_COMBOS: usize = 1326;

pub const RANK_CHARS: [char; 13] = [
    '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A',
];
pub const SUIT_CHARS: [char; 4] = ['c', 'd', 'h', 's'];

#[inline(always)]
pub fn rank(c: Card) -> u8 {
    c >> 2
}

#[inline(always)]
pub fn suit(c: Card) -> u8 {
    c & 3
}

#[inline(always)]
pub fn make_card(rank: u8, suit: u8) -> Card {
    (rank << 2) | suit
}

#[inline(always)]
pub fn card_mask(c: Card) -> u64 {
    1u64 << c
}

pub fn rank_from_char(ch: char) -> Option<u8> {
    let ch = ch.to_ascii_uppercase();
    RANK_CHARS.iter().position(|&r| r == ch).map(|i| i as u8)
}

pub fn suit_from_char(ch: char) -> Option<u8> {
    let ch = ch.to_ascii_lowercase();
    SUIT_CHARS.iter().position(|&s| s == ch).map(|i| i as u8)
}

pub fn card_from_str(s: &str) -> Result<Card, String> {
    let chars: Vec<char> = s.trim().chars().collect();
    if chars.len() != 2 {
        return Err(format!("invalid card: {s:?}"));
    }
    let r = rank_from_char(chars[0]).ok_or_else(|| format!("invalid rank in card: {s:?}"))?;
    let su = suit_from_char(chars[1]).ok_or_else(|| format!("invalid suit in card: {s:?}"))?;
    Ok(make_card(r, su))
}

pub fn card_to_string(c: Card) -> String {
    format!(
        "{}{}",
        RANK_CHARS[rank(c) as usize],
        SUIT_CHARS[suit(c) as usize]
    )
}

/// Parse a board string like "As Kh 7d", "AsKh7d", or "As,Kh,7d".
pub fn parse_cards(s: &str) -> Result<Vec<Card>, String> {
    let cleaned: String = s
        .chars()
        .filter(|c| !c.is_whitespace() && *c != ',')
        .collect();
    if cleaned.len() % 2 != 0 {
        return Err(format!("invalid card list: {s:?}"));
    }
    let chars: Vec<char> = cleaned.chars().collect();
    let mut out = Vec::with_capacity(chars.len() / 2);
    let mut seen = 0u64;
    for pair in chars.chunks(2) {
        let card = card_from_str(&pair.iter().collect::<String>())?;
        if seen & card_mask(card) != 0 {
            return Err(format!("duplicate card in list: {s:?}"));
        }
        seen |= card_mask(card);
        out.push(card);
    }
    Ok(out)
}

pub fn cards_to_string(cards: &[Card]) -> String {
    cards.iter().map(|&c| card_to_string(c)).collect()
}

/// Apply a suit permutation (index = old suit, value = new suit) to a card.
#[inline(always)]
pub fn permute_card(c: Card, pm: &[u8; 4]) -> Card {
    make_card(rank(c), pm[suit(c) as usize])
}

/// Canonical index of an unordered two-card combo. Requires `c1 != c2`.
#[inline(always)]
pub fn combo_index(c1: Card, c2: Card) -> usize {
    let (hi, lo) = if c1 > c2 { (c1, c2) } else { (c2, c1) };
    (hi as usize * (hi as usize - 1)) / 2 + lo as usize
}

/// Inverse of `combo_index`: returns (high card, low card).
pub fn combo_from_index(idx: usize) -> (Card, Card) {
    // hi is the largest h with h*(h-1)/2 <= idx
    let mut hi = 1usize;
    while (hi + 1) * hi / 2 <= idx {
        hi += 1;
    }
    let lo = idx - hi * (hi - 1) / 2;
    (hi as Card, lo as Card)
}

pub fn combo_to_string(c1: Card, c2: Card) -> String {
    let (hi, lo) = if c1 > c2 { (c1, c2) } else { (c2, c1) };
    // Display higher rank first; on equal ranks higher suit first.
    let (a, b) = if rank(hi) >= rank(lo) { (hi, lo) } else { (lo, hi) };
    format!("{}{}", card_to_string(a), card_to_string(b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn card_roundtrip() {
        for c in 0..52u8 {
            assert_eq!(card_from_str(&card_to_string(c)).unwrap(), c);
        }
    }

    #[test]
    fn combo_index_roundtrip() {
        let mut seen = std::collections::HashSet::new();
        for c1 in 0..52u8 {
            for c2 in 0..c1 {
                let idx = combo_index(c1, c2);
                assert!(idx < NUM_COMBOS);
                assert!(seen.insert(idx), "duplicate combo index");
                assert_eq!(combo_from_index(idx), (c1, c2));
            }
        }
        assert_eq!(seen.len(), NUM_COMBOS);
    }

    #[test]
    fn parse_board() {
        let b = parse_cards("As Kh 7d").unwrap();
        assert_eq!(b.len(), 3);
        assert_eq!(cards_to_string(&b), "AsKh7d");
        assert!(parse_cards("AsAs").is_err());
        assert!(parse_cards("Xx").is_err());
    }
}

/// All strategically distinct flops (suit-isomorphism classes): 1755
/// classes covering the C(52,3) = 22,100 raw flops. Canonical
/// representative = the lexicographically smallest suit-permutation of the
/// descending-sorted three cards; weight = raw flops in the class. Returned
/// high-to-low in canonical card order, deterministic.
/// Deterministic weighted systematic subset of the canonical flops: n
/// boards spread across the weight distribution (texture-proportional).
pub fn canonical_flops_subset(n: usize) -> Vec<(String, u32)> {
    let all = canonical_flops();
    if n == 0 || n >= all.len() {
        return all;
    }
    let total: f64 = all.iter().map(|x| x.1 as f64).sum();
    let mut out = Vec::with_capacity(n);
    let (mut cum, mut ti) = (0f64, 0usize);
    for (b, w) in &all {
        cum += *w as f64;
        while ti < n && ((ti as f64 + 0.5) / n as f64) * total <= cum {
            out.push((b.clone(), *w));
            ti += 1;
        }
    }
    out
}

pub fn canonical_flops() -> Vec<(String, u32)> {
    let mut perms: Vec<[u8; 4]> = Vec::with_capacity(24);
    for a in 0..4u8 {
        for b in 0..4u8 {
            if b == a {
                continue;
            }
            for c in 0..4u8 {
                if c == a || c == b {
                    continue;
                }
                perms.push([a, b, c, 6 - a - b - c]);
            }
        }
    }
    let mut classes: std::collections::BTreeMap<[u8; 3], u32> = std::collections::BTreeMap::new();
    for x in 0..52u8 {
        for y in 0..x {
            for z in 0..y {
                let mut best = [255u8; 3];
                for p in &perms {
                    let mut t = [
                        make_card(rank(x), p[suit(x) as usize]),
                        make_card(rank(y), p[suit(y) as usize]),
                        make_card(rank(z), p[suit(z) as usize]),
                    ];
                    t.sort_unstable_by(|a, b| b.cmp(a));
                    if t < best {
                        best = t;
                    }
                }
                *classes.entry(best).or_insert(0) += 1;
            }
        }
    }
    classes
        .into_iter()
        .rev()
        .map(|(k, w)| {
            (
                format!(
                    "{}{}{}",
                    card_to_string(k[0]),
                    card_to_string(k[1]),
                    card_to_string(k[2])
                ),
                w,
            )
        })
        .collect()
}
