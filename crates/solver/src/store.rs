//! Solver data storage: plain f32 arenas, or compressed
//! 16-bit arenas with one scale factor per node.
//!
//! Compressed mode stores cumulative regrets as i16 and cumulative strategy
//! as u16. Each node's block is quantized against the block's max magnitude,
//! kept in a per-node side array ("per-node scaling"). This halves solver
//! memory, and because CFR on large trees is memory-bandwidth-bound it also
//! speeds up iteration.

use serde::{Deserialize, Serialize};

/// A flat buffer that allows unsafe disjoint mutable access from multiple
/// threads. Safety contract: traversal visits each tree node at most once per
/// pass, and node data ranges never overlap, so concurrent chance-branch
/// recursion touches disjoint slices.
pub struct ArenaBuf<T> {
    buf: Box<[T]>,
    ptr: *mut T,
}

unsafe impl<T: Send> Send for ArenaBuf<T> {}
unsafe impl<T: Sync> Sync for ArenaBuf<T> {}

impl<T: Copy + Default> ArenaBuf<T> {
    pub fn new(len: usize) -> Self {
        let mut buf = vec![T::default(); len].into_boxed_slice();
        let ptr = buf.as_mut_ptr();
        ArenaBuf { buf, ptr }
    }

    #[inline(always)]
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn slice(&self, offset: u64, len: usize) -> &mut [T] {
        debug_assert!(offset as usize + len <= self.buf.len());
        std::slice::from_raw_parts_mut(self.ptr.add(offset as usize), len)
    }

    #[inline(always)]
    pub unsafe fn read_at(&self, idx: usize) -> T {
        debug_assert!(idx < self.buf.len());
        *self.ptr.add(idx)
    }

    #[inline(always)]
    pub unsafe fn write_at(&self, idx: usize, v: T) {
        debug_assert!(idx < self.buf.len());
        *self.ptr.add(idx) = v;
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Read-only view. Only call while no traversal is mutating the buffer.
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.buf.len()) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.buf.len()) }
    }
}

/// How solver arenas are stored in memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Storage {
    /// Full-precision f32: 4 bytes per entry.
    F32,
    /// 16-bit quantized entries with one f32 scale per node:
    /// regrets as i16, strategy sums as u16. 2 bytes per entry.
    Compressed,
}

/// One solver data arena (regrets, strategy sums, or PCFR+ predictions).
pub enum Store {
    F32(ArenaBuf<f32>),
    /// Signed quantization (regrets, predictions): v = q * scale / 32767.
    I16 {
        q: ArenaBuf<i16>,
        /// Block max magnitude, indexed by tree node index.
        scale: ArenaBuf<f32>,
    },
    /// Unsigned quantization (strategy sums, always >= 0): v = q * scale / 65535.
    U16 {
        q: ArenaBuf<u16>,
        scale: ArenaBuf<f32>,
    },
}

impl Store {
    pub fn f32(len: u64) -> Store {
        Store::F32(ArenaBuf::new(len as usize))
    }

    pub fn i16(len: u64, num_nodes: usize) -> Store {
        Store::I16 {
            q: ArenaBuf::new(len as usize),
            scale: ArenaBuf::new(num_nodes),
        }
    }

    pub fn u16(len: u64, num_nodes: usize) -> Store {
        Store::U16 {
            q: ArenaBuf::new(len as usize),
            scale: ArenaBuf::new(num_nodes),
        }
    }

    pub fn bytes(&self) -> u64 {
        match self {
            Store::F32(b) => b.len() as u64 * 4,
            Store::I16 { q, scale } => q.len() as u64 * 2 + scale.len() as u64 * 4,
            Store::U16 { q, scale } => q.len() as u64 * 2 + scale.len() as u64 * 4,
        }
    }

    /// Decode `len` entries at `offset` into `out` (f32 stores copy directly).
    /// Safety: same disjointness contract as `ArenaBuf::slice`.
    #[inline]
    pub unsafe fn read_f32(&self, node_idx: u32, offset: u64, len: usize, out: &mut [f32]) {
        match self {
            Store::F32(b) => out.copy_from_slice(b.slice(offset, len)),
            Store::I16 { q, scale } => {
                decode_i16(q.slice(offset, len), scale.read_at(node_idx as usize), out)
            }
            Store::U16 { q, scale } => {
                decode_u16(q.slice(offset, len), scale.read_at(node_idx as usize), out)
            }
        }
    }

    /// Encode `src` into `len` entries at `offset`, refreshing the node scale.
    /// Safety: same disjointness contract as `ArenaBuf::slice`.
    #[inline]
    pub unsafe fn write_f32(&self, node_idx: u32, offset: u64, len: usize, src: &[f32]) {
        match self {
            Store::F32(b) => b.slice(offset, len).copy_from_slice(src),
            Store::I16 { q, scale } => {
                scale.write_at(node_idx as usize, encode_i16(src, q.slice(offset, len)))
            }
            Store::U16 { q, scale } => {
                scale.write_at(node_idx as usize, encode_u16(src, q.slice(offset, len)))
            }
        }
    }
}

#[inline]
pub fn decode_i16(q: &[i16], scale: f32, out: &mut [f32]) {
    let k = scale / 32767.0;
    for (o, &v) in out.iter_mut().zip(q.iter()) {
        *o = v as f32 * k;
    }
}

#[inline]
pub fn decode_u16(q: &[u16], scale: f32, out: &mut [f32]) {
    let k = scale / 65535.0;
    for (o, &v) in out.iter_mut().zip(q.iter()) {
        *o = v as f32 * k;
    }
}

/// Quantize `src` into i16 against a caller-provided block max, returning the
/// scale actually stored (0 for a zero or non-finite max => all-zero block).
#[inline]
pub fn quantize_i16(src: &[f32], max: f32, dst: &mut [i16]) -> f32 {
    if !(max > 0.0 && max.is_finite()) {
        dst.fill(0);
        return 0.0;
    }
    let k = 32767.0 / max;
    for (d, &v) in dst.iter_mut().zip(src.iter()) {
        *d = (v * k).round_ties_even() as i16; // `as` saturates
    }
    max
}

/// Quantize non-negative `src` into u16 against a caller-provided block max.
#[inline]
pub fn quantize_u16(src: &[f32], max: f32, dst: &mut [u16]) -> f32 {
    if !(max > 0.0 && max.is_finite()) {
        dst.fill(0);
        return 0.0;
    }
    let k = 65535.0 / max;
    for (d, &v) in dst.iter_mut().zip(src.iter()) {
        *d = (v.max(0.0) * k).round_ties_even() as u16;
    }
    max
}

/// Quantize `src` into i16, returning the scale (block max |v|).
#[inline]
pub fn encode_i16(src: &[f32], dst: &mut [i16]) -> f32 {
    let mut m = 0f32;
    for &v in src {
        m = m.max(v.abs());
    }
    quantize_i16(src, m, dst)
}

/// Quantize non-negative `src` into u16, returning the scale (block max).
#[inline]
pub fn encode_u16(src: &[f32], dst: &mut [u16]) -> f32 {
    let mut m = 0f32;
    for &v in src {
        m = m.max(v);
    }
    quantize_u16(src, m, dst)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn i16_roundtrip_accuracy() {
        let src: Vec<f32> = (0..1000)
            .map(|i| ((i as f32 * 0.7).sin() * 1500.0) - 200.0)
            .collect();
        let mut q = vec![0i16; src.len()];
        let scale = encode_i16(&src, &mut q);
        let mut back = vec![0f32; src.len()];
        decode_i16(&q, scale, &mut back);
        let tol = scale / 32767.0 * 0.5 + 1e-6;
        for (a, b) in src.iter().zip(back.iter()) {
            assert!((a - b).abs() <= tol, "roundtrip error {a} vs {b} (tol {tol})");
        }
        // re-encoding the decoded values is exactly stable
        let mut q2 = vec![0i16; src.len()];
        let scale2 = encode_i16(&back, &mut q2);
        assert_eq!(scale, scale2);
        assert_eq!(q, q2);
    }

    #[test]
    fn u16_roundtrip_accuracy() {
        let src: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.37).cos().abs() * 9.5).collect();
        let mut q = vec![0u16; src.len()];
        let scale = encode_u16(&src, &mut q);
        let mut back = vec![0f32; src.len()];
        decode_u16(&q, scale, &mut back);
        let tol = scale / 65535.0 * 0.5 + 1e-6;
        for (a, b) in src.iter().zip(back.iter()) {
            assert!((a - b).abs() <= tol, "roundtrip error {a} vs {b} (tol {tol})");
        }
    }

    #[test]
    fn zero_and_nonfinite_blocks() {
        let mut q = vec![7i16; 4];
        assert_eq!(encode_i16(&[0.0; 4], &mut q), 0.0);
        assert_eq!(q, vec![0; 4]);
        let mut q = vec![7i16; 2];
        assert_eq!(encode_i16(&[f32::NAN, f32::INFINITY], &mut q), 0.0);
        assert_eq!(q, vec![0; 2]);
    }
}
