//! Save/load of solved spots: a JSON header line followed by raw f32 arenas.
//!
//! The on-disk format is always full-precision f32 regardless of the in-RAM
//! storage mode, so files written by compressed and uncompressed solvers are
//! interchangeable (and older saves keep loading).

use crate::cfr::Solver;
use crate::game::{Spot, SpotConfig};
use crate::store::{Storage, Store};
use crate::tree::KIND_ACTION;
use serde::{Deserialize, Serialize};
use std::io::{BufReader, BufWriter, Read, Write};
use std::sync::Arc;

const MAGIC: &[u8] = b"GTOSOLVE2\n";

#[derive(Serialize, Deserialize)]
struct Header {
    config: SpotConfig,
    iteration: u32,
    // Locked sigmas live only in the solver's maps (CFR freezes the arenas at
    // locked nodes), so they must ride along. serde defaults keep pre-lock
    // saves loading, and old readers ignore the extra fields.
    #[serde(default)]
    locks: Vec<(u32, Vec<f32>)>,
    #[serde(default)]
    lock_labels: Vec<(u32, String)>,
}

/// Validate saved lock entries against the tree a [`Spot`] built.
///
/// Every consumer of `Solver::locks` assumes the exact shape — `sigma` copied
/// with `copy_from_slice`, indexed unchecked as `l[a * nh + j]`, the node
/// index used to address per-node offset tables — so a malformed entry that
/// reaches a live solver panics at the first query or solve step, under the
/// session mutex. Checks, per `(node_idx, sigma)` entry: the index is in
/// range, the node is an action node, `sigma.len()` is exactly
/// `num_children * num_hands(actor)`, and every value is finite and >= 0.
///
/// `load_with_storage` runs this before installing the locks, but callers
/// that vet a save file BEFORE discarding existing state (the server's
/// peek-then-swap load path) should also run it on the peeked header's lock
/// entries with the rebuilt spot, since a load-time error still fires after
/// the old session is gone.
pub fn validate_locks(spot: &Spot, locks: &[(u32, Vec<f32>)]) -> Result<(), String> {
    for (idx, sigma) in locks {
        let node = spot.tree.nodes.get(*idx as usize).ok_or_else(|| {
            format!(
                "lock at node {idx}: index out of range (tree has {} nodes)",
                spot.tree.nodes.len()
            )
        })?;
        if node.kind != KIND_ACTION {
            return Err(format!("lock at node {idx}: not an action node"));
        }
        let na = node.num_children as usize;
        let nh = spot.hands[node.player as usize].len();
        if sigma.len() != na * nh {
            return Err(format!(
                "lock at node {idx}: sigma has {} entries, expected {na} actions x {nh} hands = {}",
                sigma.len(),
                na * nh
            ));
        }
        if let Some(v) = sigma.iter().find(|v| !v.is_finite() || **v < 0.0) {
            return Err(format!(
                "lock at node {idx}: invalid frequency {v} (must be finite and >= 0)"
            ));
        }
    }
    Ok(())
}

impl Solver {
    /// Decode one whole arena (player `p`'s action-node blocks) to f32.
    pub(crate) fn arena_to_f32(&self, store: &Store, p: usize) -> Vec<f32> {
        let len = self.spot.tree.data_size[p] as usize;
        let nh = self.spot.hands[p].len();
        let mut buf = vec![0f32; len];
        for (idx, node) in self.spot.tree.nodes.iter().enumerate() {
            if node.kind == KIND_ACTION && node.player as usize == p {
                let n = node.num_children as usize * nh;
                let off = node.data_offset as usize;
                unsafe {
                    store.read_f32(idx as u32, node.data_offset, n, &mut buf[off..off + n]);
                }
            }
        }
        buf
    }

    pub fn save(&self, path: &str) -> Result<(), String> {
        let file = std::fs::File::create(path).map_err(|e| e.to_string())?;
        let mut w = BufWriter::new(file);
        w.write_all(MAGIC).map_err(|e| e.to_string())?;
        let header = Header {
            config: self.spot.config.clone(),
            iteration: self.iteration,
            locks: self.locks.iter().map(|(k, v)| (*k, v.clone())).collect(),
            lock_labels: self.lock_labels.iter().map(|(k, v)| (*k, v.clone())).collect(),
        };
        let hjson = serde_json::to_string(&header).map_err(|e| e.to_string())?;
        w.write_all(hjson.as_bytes()).map_err(|e| e.to_string())?;
        w.write_all(b"\n").map_err(|e| e.to_string())?;
        for (store, p) in [
            (&self.regrets[0], 0usize),
            (&self.regrets[1], 1),
            (&self.strat[0], 0),
            (&self.strat[1], 1),
        ] {
            let write_slice = |w: &mut BufWriter<std::fs::File>, slice: &[f32]| {
                w.write_all(&(slice.len() as u64).to_le_bytes())
                    .map_err(|e| e.to_string())?;
                // f32 slice as raw little-endian bytes
                let bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len() * 4)
                };
                w.write_all(bytes).map_err(|e| e.to_string())
            };
            match store {
                Store::F32(b) => write_slice(&mut w, b.as_slice())?,
                _ => write_slice(&mut w, &self.arena_to_f32(store, p))?,
            }
        }
        w.flush().map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Solver, String> {
        Solver::load_with_storage(path, Storage::F32)
    }

    pub fn load_with_storage(path: &str, storage: Storage) -> Result<Solver, String> {
        let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
        let mut r = BufReader::new(file);
        let mut magic = [0u8; 10];
        r.read_exact(&mut magic).map_err(|e| e.to_string())?;
        if magic != MAGIC {
            return Err("not a solver save file".to_string());
        }
        let mut header_line = Vec::new();
        loop {
            let mut b = [0u8; 1];
            r.read_exact(&mut b).map_err(|e| e.to_string())?;
            if b[0] == b'\n' {
                break;
            }
            header_line.push(b[0]);
        }
        let header: Header =
            serde_json::from_slice(&header_line).map_err(|e| format!("bad header: {e}"))?;
        // Lenient: a saved config may carry sizing quirks the pre-validation
        // builder silently dropped; the arenas match that dropped-size tree,
        // so the rebuild must reproduce it rather than reject the config.
        let spot = Spot::new_lenient(header.config)?;
        // Refuse malformed lock entries here, while no state depends on them:
        // installed unchecked they would panic at the first query instead.
        validate_locks(&spot, &header.locks).map_err(|e| format!("bad lock section: {e}"))?;
        let mut solver = Solver::with_storage(Arc::new(spot), storage);
        solver.iteration = header.iteration;
        solver.locks = header.locks.into_iter().collect();
        solver.lock_labels = header.lock_labels.into_iter().collect();
        if !solver.locks.is_empty() {
            // sibling branches must re-materialize the locked sigmas
            solver.mark_sym_dirty();
        }
        for arena in [0usize, 1, 2, 3] {
            let p = arena % 2;
            let mut len_bytes = [0u8; 8];
            r.read_exact(&mut len_bytes).map_err(|e| e.to_string())?;
            let len = u64::from_le_bytes(len_bytes) as usize;
            let expected = solver.spot.tree.data_size[p] as usize;
            if len != expected {
                return Err(format!(
                    "arena size mismatch: file {len}, expected {expected} (tree config changed?)"
                ));
            }
            let mut buf = vec![0f32; len];
            {
                let bytes: &mut [u8] = unsafe {
                    std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, len * 4)
                };
                r.read_exact(bytes).map_err(|e| e.to_string())?;
            }
            let store = match arena {
                0 | 1 => &solver.regrets[p],
                _ => &solver.strat[p],
            };
            match store {
                Store::F32(b) => unsafe { b.slice(0, len) }.copy_from_slice(&buf),
                _ => {
                    let nh = solver.spot.hands[p].len();
                    for (idx, node) in solver.spot.tree.nodes.iter().enumerate() {
                        if node.kind == KIND_ACTION && node.player as usize == p {
                            let n = node.num_children as usize * nh;
                            let off = node.data_offset as usize;
                            unsafe {
                                store.write_f32(
                                    idx as u32,
                                    node.data_offset,
                                    n,
                                    &buf[off..off + n],
                                );
                            }
                        }
                    }
                }
            }
        }
        Ok(solver)
    }
}
