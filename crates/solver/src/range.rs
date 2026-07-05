//! Range parsing and formatting (standard solver text syntax: AA, ATs+, QQ:0.5).
//!
//! Supported tokens (comma or whitespace separated, each with optional `:weight`):
//! - Pairs: `AA`, `TT+`, `99-66`
//! - Suited/offsuit/both: `AKs`, `AKo`, `AK`, `ATs+`, `A5s-A2s`, `T9s-54s` (same-gap run)
//! - Specific combos: `AhKh`, `AsAd`
//! - Weights: `AA:0.5` or `AA:50` (values > 1 are treated as percentages)
//!
//! Later tokens overwrite earlier ones for the combos they cover.

use crate::cards::*;

/// A range is a weight in [0, 1] for each of the 1326 combos.
#[derive(Clone)]
pub struct Range {
    pub weights: Vec<f32>, // len NUM_COMBOS, indexed by combo_index
}

impl Default for Range {
    fn default() -> Self {
        Range {
            weights: vec![0.0; NUM_COMBOS],
        }
    }
}

impl Range {
    pub fn parse(s: &str) -> Result<Range, String> {
        let mut range = Range::default();
        let normalized = s.replace(['\n', '\t', ';'], ",");
        for raw_token in normalized.split([',', ' ']) {
            let token = raw_token.trim();
            if token.is_empty() {
                continue;
            }
            let (item, weight) = match token.rsplit_once(':') {
                Some((item, w)) => {
                    let mut weight: f32 = w
                        .trim()
                        .parse()
                        .map_err(|_| format!("invalid weight in token {token:?}"))?;
                    if weight > 1.0 {
                        weight /= 100.0;
                    }
                    if !(0.0..=1.0).contains(&weight) {
                        return Err(format!("weight out of range in token {token:?}"));
                    }
                    (item.trim(), weight)
                }
                None => (token, 1.0),
            };
            let combos = expand_item(item)?;
            for (c1, c2) in combos {
                range.weights[combo_index(c1, c2)] = weight;
            }
        }
        Ok(range)
    }

    pub fn is_empty(&self) -> bool {
        self.weights.iter().all(|&w| w <= 0.0)
    }

    /// Total number of (weighted) combos.
    pub fn num_combos(&self) -> f32 {
        self.weights.iter().sum()
    }

    /// Compact string representation, grouping uniform hand classes.
    pub fn to_string_compact(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        // Iterate over the 169 classes; if all present combos in the class share
        // one weight, emit the class; otherwise emit per-combo tokens.
        for hi in (0..13u8).rev() {
            for lo in (0..=hi).rev() {
                for &suitedness in &[Suitedness::Pair, Suitedness::Suited, Suitedness::Offsuit] {
                    let class_combos = class_combo_list(hi, lo, suitedness);
                    if class_combos.is_empty() {
                        continue;
                    }
                    let weights: Vec<f32> = class_combos
                        .iter()
                        .map(|&(a, b)| self.weights[combo_index(a, b)])
                        .collect();
                    let nonzero: Vec<f32> =
                        weights.iter().copied().filter(|&w| w > 0.0).collect();
                    if nonzero.is_empty() {
                        continue;
                    }
                    let label = class_label(hi, lo, suitedness);
                    let uniform = nonzero.len() == weights.len()
                        && nonzero.iter().all(|&w| (w - nonzero[0]).abs() < 1e-4);
                    if uniform {
                        if (nonzero[0] - 1.0).abs() < 1e-4 {
                            parts.push(label);
                        } else {
                            parts.push(format!("{}:{}", label, fmt_weight(nonzero[0])));
                        }
                    } else {
                        for (&(a, b), &w) in class_combos.iter().zip(weights.iter()) {
                            if w > 0.0 {
                                let combo = combo_to_string(a, b);
                                if (w - 1.0).abs() < 1e-4 {
                                    parts.push(combo);
                                } else {
                                    parts.push(format!("{}:{}", combo, fmt_weight(w)));
                                }
                            }
                        }
                    }
                }
            }
        }
        parts.join(",")
    }
}

fn fmt_weight(w: f32) -> String {
    let s = format!("{:.3}", w);
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Suitedness {
    Pair,
    Suited,
    Offsuit,
}

fn class_label(hi: u8, lo: u8, s: Suitedness) -> String {
    let h = RANK_CHARS[hi as usize];
    let l = RANK_CHARS[lo as usize];
    match s {
        Suitedness::Pair => format!("{h}{l}"),
        Suitedness::Suited => format!("{h}{l}s"),
        Suitedness::Offsuit => format!("{h}{l}o"),
    }
}

/// All combos of a hand class. Empty if class is invalid (e.g. suited pair).
pub fn class_combo_list(hi: u8, lo: u8, s: Suitedness) -> Vec<(Card, Card)> {
    let mut out = Vec::new();
    match s {
        Suitedness::Pair => {
            if hi != lo {
                return out;
            }
            for s1 in 0..4u8 {
                for s2 in 0..s1 {
                    out.push((make_card(hi, s1), make_card(lo, s2)));
                }
            }
        }
        Suitedness::Suited => {
            if hi == lo {
                return out;
            }
            for su in 0..4u8 {
                out.push((make_card(hi, su), make_card(lo, su)));
            }
        }
        Suitedness::Offsuit => {
            if hi == lo {
                return out;
            }
            for s1 in 0..4u8 {
                for s2 in 0..4u8 {
                    if s1 != s2 {
                        out.push((make_card(hi, s1), make_card(lo, s2)));
                    }
                }
            }
        }
    }
    out
}

fn expand_class(hi: u8, lo: u8, marker: Option<char>) -> Result<Vec<(Card, Card)>, String> {
    let mut out = Vec::new();
    if hi == lo {
        if marker.is_some() {
            return Err(format!(
                "pairs cannot have a suitedness marker: {}{}",
                RANK_CHARS[hi as usize], RANK_CHARS[lo as usize]
            ));
        }
        out.extend(class_combo_list(hi, lo, Suitedness::Pair));
    } else {
        match marker {
            Some('s') => out.extend(class_combo_list(hi, lo, Suitedness::Suited)),
            Some('o') => out.extend(class_combo_list(hi, lo, Suitedness::Offsuit)),
            None => {
                out.extend(class_combo_list(hi, lo, Suitedness::Suited));
                out.extend(class_combo_list(hi, lo, Suitedness::Offsuit));
            }
            Some(c) => return Err(format!("invalid suitedness marker {c:?}")),
        }
    }
    Ok(out)
}

/// Parse one hand-class spec like "AKs" / "TT" -> (hi, lo, marker).
fn parse_class(s: &str) -> Option<(u8, u8, Option<char>)> {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() < 2 || chars.len() > 3 {
        return None;
    }
    let r1 = rank_from_char(chars[0])?;
    let r2 = rank_from_char(chars[1])?;
    let marker = if chars.len() == 3 {
        let m = chars[2].to_ascii_lowercase();
        if m != 's' && m != 'o' {
            return None;
        }
        Some(m)
    } else {
        None
    };
    let (hi, lo) = if r1 >= r2 { (r1, r2) } else { (r2, r1) };
    Some((hi, lo, marker))
}

fn expand_item(item: &str) -> Result<Vec<(Card, Card)>, String> {
    let item = item.trim();

    // Specific combo: "AhKh"
    if item.len() == 4 {
        let chars: Vec<char> = item.chars().collect();
        if rank_from_char(chars[0]).is_some()
            && suit_from_char(chars[1]).is_some()
            && rank_from_char(chars[2]).is_some()
            && suit_from_char(chars[3]).is_some()
        {
            let c1 = card_from_str(&item[0..2])?;
            let c2 = card_from_str(&item[2..4])?;
            if c1 == c2 {
                return Err(format!("combo with duplicate card: {item:?}"));
            }
            return Ok(vec![(c1, c2)]);
        }
    }

    // Range: "99-66", "AQs-A9s", "T9s-54s"
    if let Some((a, b)) = item.split_once('-') {
        let (h1, l1, m1) =
            parse_class(a.trim()).ok_or_else(|| format!("invalid range start: {item:?}"))?;
        let (h2, l2, m2) =
            parse_class(b.trim()).ok_or_else(|| format!("invalid range end: {item:?}"))?;
        if m1 != m2 {
            return Err(format!("mismatched suitedness in range: {item:?}"));
        }
        let mut out = Vec::new();
        if h1 == l1 && h2 == l2 {
            // pair range
            let (top, bot) = if h1 >= h2 { (h1, h2) } else { (h2, h1) };
            for r in bot..=top {
                out.extend(expand_class(r, r, m1)?);
            }
        } else if h1 == h2 {
            // same high card, vary low: AQs-A9s
            let (top, bot) = if l1 >= l2 { (l1, l2) } else { (l2, l1) };
            for r in bot..=top {
                if r == h1 {
                    continue;
                }
                out.extend(expand_class(h1, r, m1)?);
            }
        } else if h1 as i32 - l1 as i32 == h2 as i32 - l2 as i32 {
            // same-gap run: T9s-54s
            let (top, bot) = if h1 >= h2 { (h1, h2) } else { (h2, h1) };
            let gap = h1 - l1;
            for h in bot..=top {
                out.extend(expand_class(h, h - gap, m1)?);
            }
        } else {
            return Err(format!("unsupported range shape: {item:?}"));
        }
        return Ok(out);
    }

    // Plus: "TT+", "ATs+", "AT+"
    if let Some(base) = item.strip_suffix('+') {
        let (hi, lo, marker) =
            parse_class(base.trim()).ok_or_else(|| format!("invalid item: {item:?}"))?;
        let mut out = Vec::new();
        if hi == lo {
            for r in hi..13 {
                out.extend(expand_class(r, r, marker)?);
            }
        } else {
            // Fixed high card, low card increments up to one below high.
            for l in lo..hi {
                out.extend(expand_class(hi, l, marker)?);
            }
        }
        return Ok(out);
    }

    // Plain class
    let (hi, lo, marker) =
        parse_class(item).ok_or_else(|| format!("invalid range token: {item:?}"))?;
    expand_class(hi, lo, marker)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn count(s: &str) -> f32 {
        Range::parse(s).unwrap().num_combos()
    }

    #[test]
    fn combo_counts() {
        assert_eq!(count("AA"), 6.0);
        assert_eq!(count("AKs"), 4.0);
        assert_eq!(count("AKo"), 12.0);
        assert_eq!(count("AK"), 16.0);
        assert_eq!(count("AhKh"), 1.0);
        assert_eq!(count("TT+"), 30.0); // TT JJ QQ KK AA
        assert_eq!(count("ATs+"), 16.0); // ATs AJs AQs AKs
        assert_eq!(count("99-66"), 24.0);
        assert_eq!(count("A5s-A2s"), 16.0);
        assert_eq!(count("T9s-87s"), 12.0); // T9s 98s 87s
        assert_eq!(count("AA,KK,QQ"), 18.0);
    }

    #[test]
    fn weights() {
        let r = Range::parse("AA:0.5,KK").unwrap();
        let (a1, a2) = (make_card(12, 0), make_card(12, 1));
        let (k1, k2) = (make_card(11, 0), make_card(11, 1));
        assert!((r.weights[combo_index(a1, a2)] - 0.5).abs() < 1e-6);
        assert!((r.weights[combo_index(k1, k2)] - 1.0).abs() < 1e-6);
        // percentage form
        let r2 = Range::parse("AA:50").unwrap();
        assert!((r2.weights[combo_index(a1, a2)] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn overwrite_later_tokens() {
        let r = Range::parse("AA,AA:0.25").unwrap();
        let (a1, a2) = (make_card(12, 0), make_card(12, 1));
        assert!((r.weights[combo_index(a1, a2)] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn roundtrip_compact() {
        let src = "AA,KK:0.5,AKs,A5s-A2s,KQo:0.3,76s,AhQh:0.77";
        let r = Range::parse(src).unwrap();
        let s = r.to_string_compact();
        let r2 = Range::parse(&s).unwrap();
        for i in 0..NUM_COMBOS {
            assert!(
                (r.weights[i] - r2.weights[i]).abs() < 1e-3,
                "mismatch at {i}: {} vs {} ({s})",
                r.weights[i],
                r2.weights[i]
            );
        }
    }

    #[test]
    fn rejects_garbage() {
        assert!(Range::parse("XX").is_err());
        assert!(Range::parse("AAs").is_err());
        assert!(Range::parse("AKs:abc").is_err());
    }

    #[test]
    fn weight_above_one_is_percent() {
        let r = Range::parse("AKs:1.5").unwrap(); // interpreted as 1.5%
        let c1 = make_card(12, 0);
        let c2 = make_card(11, 0);
        assert!((r.weights[combo_index(c1, c2)] - 0.015).abs() < 1e-6);
    }
}
