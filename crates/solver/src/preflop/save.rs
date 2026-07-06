//! Save/load of preflop games: a JSON header line (config, iteration, seat
//! models, point locks) followed by the raw f32 regret and strategy-sum
//! arenas. The equity table is NOT stored — it is deterministic and
//! disk-cached separately; the tree is rebuilt from the config on load and
//! must produce identical arena sizes (the builder is deterministic).

use super::equity::EquityTable;
use super::{PreflopConfig, PreflopSolver, SeatProfile};
use serde::{Deserialize, Serialize};
use std::io::{BufReader, BufWriter, Read, Write};
use std::sync::Arc;

const MAGIC: &[u8] = b"GTOPREFLOP1\n";

#[derive(Serialize, Deserialize)]
struct Header {
    config: PreflopConfig,
    iteration: u32,
    seat_frozen: Vec<bool>,
    seat_profiles: Vec<Option<SeatProfile>>,
    point_locks: Vec<(u32, Vec<f32>)>,
}

fn write_slice(w: &mut BufWriter<std::fs::File>, slice: &[f32]) -> Result<(), String> {
    w.write_all(&(slice.len() as u64).to_le_bytes())
        .map_err(|e| e.to_string())?;
    // f32 slice as raw little-endian bytes (same convention as postflop saves)
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len() * 4) };
    w.write_all(bytes).map_err(|e| e.to_string())
}

fn read_slice(r: &mut BufReader<std::fs::File>, into: &mut [f32]) -> Result<(), String> {
    let mut lenb = [0u8; 8];
    r.read_exact(&mut lenb).map_err(|e| e.to_string())?;
    let len = u64::from_le_bytes(lenb) as usize;
    if len != into.len() {
        return Err(format!(
            "arena size mismatch: file has {len} entries, rebuilt tree needs {}",
            into.len()
        ));
    }
    let bytes: &mut [u8] =
        unsafe { std::slice::from_raw_parts_mut(into.as_mut_ptr() as *mut u8, len * 4) };
    r.read_exact(bytes).map_err(|e| e.to_string())
}

impl PreflopSolver {
    pub fn save_game(&self, path: &str) -> Result<(), String> {
        let file = std::fs::File::create(path).map_err(|e| e.to_string())?;
        let mut w = BufWriter::new(file);
        w.write_all(MAGIC).map_err(|e| e.to_string())?;
        let header = Header {
            config: self.cfg.clone(),
            iteration: self.iteration,
            seat_frozen: self.seat_frozen.clone(),
            seat_profiles: self.seat_profiles.clone(),
            point_locks: self.point_locks.iter().map(|(k, v)| (*k, v.clone())).collect(),
        };
        let hjson = serde_json::to_string(&header).map_err(|e| e.to_string())?;
        w.write_all(hjson.as_bytes()).map_err(|e| e.to_string())?;
        w.write_all(b"\n").map_err(|e| e.to_string())?;
        // SAFETY: callers hold exclusive access (the server serializes solver
        // use through a mutex); no traversal mutates the arenas while we read
        unsafe {
            write_slice(&mut w, self.regrets.slice())?;
            write_slice(&mut w, self.strat_sum.slice())?;
        }
        w.flush().map_err(|e| e.to_string())
    }

    pub fn load_game(path: &str, eq: Arc<EquityTable>) -> Result<PreflopSolver, String> {
        let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
        let mut r = BufReader::new(file);
        let mut magic = [0u8; 12];
        r.read_exact(&mut magic).map_err(|e| e.to_string())?;
        if magic != MAGIC {
            return Err("not a preflop game save".to_string());
        }
        let mut line = Vec::new();
        loop {
            let mut b = [0u8; 1];
            r.read_exact(&mut b).map_err(|e| e.to_string())?;
            if b[0] == b'\n' {
                break;
            }
            line.push(b[0]);
        }
        let header: Header = serde_json::from_slice(&line).map_err(|e| e.to_string())?;
        let mut s = PreflopSolver::new(header.config, eq)?;
        if header.seat_frozen.len() != s.n || header.seat_profiles.len() != s.n {
            return Err("save is inconsistent (seat count mismatch)".to_string());
        }
        // SAFETY: `s` is exclusively ours; nothing else touches its arenas
        unsafe {
            read_slice(&mut r, s.regrets.slice_mut())?;
            read_slice(&mut r, s.strat_sum.slice_mut())?;
        }
        s.iteration = header.iteration;
        s.seat_frozen = header.seat_frozen;
        s.seat_profiles = header.seat_profiles;
        s.point_locks = header.point_locks.into_iter().collect();
        Ok(s)
    }
}
