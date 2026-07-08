//! Local web server hosting the solver and the browser UI.

use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use solver::cfr::{Algorithm, Solver};
use solver::game::{Spot, SpotConfig};
use solver::query::PathStep;
use solver::range::Range;
use solver::store::Storage;
use solver::tree::{parse_sizes, StreetSizing, TreeConfig};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

type ApiError = (StatusCode, String);

fn bad_request(msg: impl Into<String>) -> ApiError {
    (StatusCode::BAD_REQUEST, msg.into())
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

#[derive(Default)]
struct AppState {
    session: Mutex<Option<Session>>,
    status: Mutex<StatusInfo>,
    preflop: Mutex<Option<PreflopSession>>,
    report: Mutex<ReportStatus>,
    report_stop: Arc<AtomicBool>,
}

#[derive(Clone, Serialize, Default)]
struct ReportStatus {
    running: bool,
    name: String,
    done: usize,
    total: usize,
    board: String,
    error: String,
    seconds: f64,
}

struct PreflopSession {
    solver: Arc<Mutex<solver::preflop::PreflopSolver>>,
    stop: Arc<AtomicBool>,
    status: Arc<Mutex<PreflopStatus>>,
}

#[derive(Clone, Serialize, Default)]
struct PreflopStatus {
    /// "idle" | "running" | "done" | "stopped"
    state: String,
    /// While running: "iterating" or "measuring" (the best-response
    /// accuracy pass — long on big trees; the UI explains the pause).
    #[serde(default)]
    phase: String,
    /// True while the preflop solve runs on the GPU.
    #[serde(default)]
    gpu: bool,
    /// Why it isn't on the GPU (fallback reason), when applicable.
    #[serde(default)]
    gpu_note: String,
    iteration: u32,
    /// Per-player best-response gaps (bb) and their sum — the convergence
    /// metric for the preflop model (multiway has no exploitability proper).
    gaps: Vec<f64>,
    gap_total: f64,
    evs: Vec<f64>,
}

struct Session {
    solver: Arc<Mutex<Solver>>,
    stop: Arc<AtomicBool>,
    worker: Option<std::thread::JoinHandle<()>>,
    /// Bumped whenever locks change so a running GPU solve can refresh.
    lock_gen: Arc<std::sync::atomic::AtomicU64>,
}

#[derive(Clone, Serialize, Default)]
struct TreeInfo {
    nodes: usize,
    action_nodes: usize,
    /// Estimated solver-arena RAM (MB) for the active storage mode.
    arena_mb: f64,
    /// Estimated VRAM (MB) to solve this spot on the GPU.
    #[serde(default)]
    vram_mb: f64,
    /// VRAM ceiling (MB); a spot above this runs on the CPU.
    #[serde(default)]
    gpu_cap_mb: u64,
    /// Whether the GPU solver is compiled in and enabled (SOLVER_GPU != 0).
    #[serde(default)]
    gpu_available: bool,
    hands_oop: usize,
    hands_ip: usize,
    board: String,
}

#[derive(Clone, Serialize, Default)]
struct StatusInfo {
    /// idle | ready | running | done | stopped
    state: String,
    /// True while the current/last solve ran on the GPU.
    #[serde(default)]
    gpu: bool,
    /// Why the solve is (not) on the GPU — set on fallback so the UI can
    /// explain it (empty when on GPU or solving on CPU by choice).
    #[serde(default)]
    gpu_note: String,
    iteration: u32,
    exploit_chips: f64,
    exploit_pct: f64,
    elapsed_secs: f64,
    history: Vec<HistoryPoint>,
    tree: Option<TreeInfo>,
    spot_request: Option<SpotRequest>,
}

#[derive(Clone, Serialize, Deserialize)]
struct HistoryPoint {
    iteration: u32,
    exploit_pct: f64,
}

// ---------------------------------------------------------------------------
// Wire formats
// ---------------------------------------------------------------------------

#[derive(Clone, Serialize, Deserialize)]
struct SizesRequest {
    bet: String,
    raise: String,
    donk: String,
}

#[derive(Clone, Serialize, Deserialize)]
struct SpotRequest {
    board: String,
    range_oop: String,
    range_ip: String,
    starting_pot: f64,
    effective_stack: f64,
    /// Percent, e.g. 5 for 5% rake.
    #[serde(default)]
    rake_pct: f64,
    #[serde(default)]
    rake_cap: f64,
    /// Percent, e.g. 85.
    #[serde(default = "default_allin_threshold")]
    allin_threshold: f64,
    #[serde(default)]
    add_allin: bool,
    #[serde(default = "default_max_raises")]
    max_raises: u8,
    /// [flop, turn, river]
    oop: Vec<SizesRequest>,
    ip: Vec<SizesRequest>,
}

fn default_allin_threshold() -> f64 {
    85.0
}
fn default_max_raises() -> u8 {
    10
}

fn convert_sizing(streets: &[SizesRequest]) -> Result<[StreetSizing; 3], String> {
    if streets.len() != 3 {
        return Err("need sizing for exactly 3 streets".to_string());
    }
    let mut out: [StreetSizing; 3] = Default::default();
    for (i, s) in streets.iter().enumerate() {
        out[i] = StreetSizing {
            bet: parse_sizes(&s.bet)?,
            raise: parse_sizes(&s.raise)?,
            donk: parse_sizes(&s.donk)?,
        };
    }
    Ok(out)
}

impl SpotRequest {
    fn to_spot_config(&self) -> Result<SpotConfig, String> {
        Ok(SpotConfig {
            board: self.board.clone(),
            range_oop: self.range_oop.clone(),
            range_ip: self.range_ip.clone(),
            tree: TreeConfig {
                starting_pot: self.starting_pot,
                effective_stack: self.effective_stack,
                rake_pct: self.rake_pct / 100.0,
                rake_cap: self.rake_cap,
                oop: convert_sizing(&self.oop)?,
                ip: convert_sizing(&self.ip)?,
                allin_threshold: self.allin_threshold / 100.0,
                add_allin: self.add_allin,
                max_raises: self.max_raises,
            },
        })
    }
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// GPU solving: on when compiled with the `gpu` feature unless SOLVER_GPU=0.
#[cfg(feature = "gpu")]
fn gpu_enabled() -> bool {
    std::env::var("SOLVER_GPU").as_deref() != Ok("0")
}

/// Safety headroom (MB) kept free on top of the VRAM estimate, for the CUDA
/// context, plan arrays and lock tables not counted in the estimate.
#[cfg(feature = "gpu")]
const GPU_MARGIN_MB: u64 = 512;

/// Effective GPU memory budget (MB) and whether the GPU is usable for this run.
/// By default the budget is the card's *live free* VRAM minus a safety margin,
/// so a spot uses as much VRAM as physically fits; SOLVER_GPU_MEM_MB overrides
/// it with a fixed manual cap. Falls back to 20 GB if VRAM can't be queried.
fn gpu_budget() -> (u64, bool) {
    let manual = std::env::var("SOLVER_GPU_MEM_MB")
        .ok()
        .and_then(|v| v.parse::<u64>().ok());
    #[cfg(feature = "gpu")]
    {
        if gpu_enabled() {
            if let Some((free_mb, _total)) = solver::gpu::vram_info_mb() {
                return (manual.unwrap_or(free_mb.saturating_sub(GPU_MARGIN_MB)), true);
            }
        }
    }
    (manual.unwrap_or(20_000), false)
}

/// Build the tree-info summary returned to the UI, including the RAM (arena)
/// and estimated VRAM footprint plus the live GPU budget for this spot.
fn tree_info(spot: &Spot, arena_mb: f64) -> TreeInfo {
    let (gpu_cap_mb, gpu_available) = gpu_budget();
    TreeInfo {
        nodes: spot.tree.nodes.len(),
        action_nodes: spot.num_action_nodes(),
        arena_mb,
        vram_mb: spot.vram_estimate_bytes() as f64 / 1e6,
        gpu_cap_mb,
        gpu_available,
        hands_oop: spot.hands[0].len(),
        hands_ip: spot.hands[1].len(),
        board: spot.config.board.clone(),
    }
}

/// Solver-arena RAM cap (MB): SOLVER_MEM_MB override, else 80% of currently
/// available system memory (never above 48 GB), so a laptop refuses a spot
/// sized for a workstation instead of thrashing into OOM.
fn mem_cap_mb() -> f64 {
    if let Some(v) = std::env::var("SOLVER_MEM_MB")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
    {
        return v;
    }
    let avail_mb = std::fs::read_to_string("/proc/meminfo").ok().and_then(|s| {
        s.lines()
            .find(|l| l.starts_with("MemAvailable:"))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|kb| kb.parse::<f64>().ok())
            .map(|kb| kb / 1024.0)
    });
    match avail_mb {
        Some(a) => (a * 0.8).min(48_000.0),
        None => 48_000.0, // no /proc/meminfo (non-Linux): keep the old cap
    }
}

/// Arena storage for new/loaded solves: compressed unless SOLVER_COMPRESS=0.
fn storage_from_env() -> Storage {
    match std::env::var("SOLVER_COMPRESS").as_deref() {
        Ok("0") => Storage::F32,
        _ => Storage::Compressed,
    }
}

async fn build_spot(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SpotRequest>,
) -> Result<Json<TreeInfo>, ApiError> {
    let config = req.to_spot_config().map_err(bad_request)?;

    // Stop any running solve and drop the old session.
    let st = state.clone();
    tokio::task::spawn_blocking(move || stop_current(&st, true))
        .await
        .map_err(|e| bad_request(e.to_string()))?;

    let spot = tokio::task::spawn_blocking(move || Spot::new(config))
        .await
        .map_err(|e| bad_request(e.to_string()))?
        .map_err(bad_request)?;

    let storage = storage_from_env();
    let arena_mb = spot.arena_bytes_for(storage) as f64 / 1e6;
    let cap_mb = mem_cap_mb();
    if arena_mb > cap_mb {
        return Err(bad_request(format!(
            "tree too large ({arena_mb:.0} MB of solver data, cap {cap_mb:.0} MB); \
             reduce bet sizes or set SOLVER_MEM_MB to override"
        )));
    }

    let info = tree_info(&spot, arena_mb);

    let solver =
        tokio::task::spawn_blocking(move || Solver::with_storage(Arc::new(spot), storage))
            .await
            .map_err(|e| bad_request(e.to_string()))?;

    *state.session.lock().unwrap() = Some(Session {
        solver: Arc::new(Mutex::new(solver)),
        stop: Arc::new(AtomicBool::new(false)),
        worker: None,
        lock_gen: Arc::new(std::sync::atomic::AtomicU64::new(0)),
    });
    let mut status = state.status.lock().unwrap();
    *status = StatusInfo {
        state: "ready".to_string(),
        tree: Some(info.clone()),
        spot_request: Some(req),
        ..Default::default()
    };
    Ok(Json(info))
}

#[derive(Deserialize)]
struct SolveRequest {
    #[serde(default = "default_max_iterations")]
    max_iterations: u32,
    #[serde(default = "default_target")]
    target_exploit_pct: f64,
    #[serde(default = "default_check_every")]
    check_every: u32,
    /// "dcfr" (default), "cfr+" or "pcfr+".
    #[serde(default)]
    algorithm: Option<String>,
}

fn default_max_iterations() -> u32 {
    2000
}
fn default_target() -> f64 {
    0.3
}
fn default_check_every() -> u32 {
    20
}

async fn start_solve(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SolveRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let mut session_guard = state.session.lock().unwrap();
    let session = session_guard
        .as_mut()
        .ok_or_else(|| bad_request("no spot built yet"))?;

    {
        let status = state.status.lock().unwrap();
        if status.state == "running" {
            return Err((StatusCode::CONFLICT, "already running".to_string()));
        }
    }

    if let Some(name) = &req.algorithm {
        let algo = Algorithm::parse(name).map_err(bad_request)?;
        session.solver.lock().unwrap().algo = algo;
    }

    session.stop.store(false, Ordering::Relaxed);
    let solver = session.solver.clone();
    let stop = session.stop.clone();
    let lock_gen = session.lock_gen.clone();
    let app = state.clone();

    {
        let mut status = state.status.lock().unwrap();
        status.state = "running".to_string();
        status.gpu = false;
        status.gpu_note = String::new();
        status.history.clear();
    }

    let handle = std::thread::spawn(move || {
        #[cfg(feature = "gpu")]
        if gpu_enabled() {
            match gpu_solve_loop(&solver, &stop, &lock_gen, &app, &req) {
                Ok(()) => return,
                Err(err) => {
                    println!("gpu solve unavailable ({err}); falling back to CPU");
                    let mut st = app.status.lock().unwrap();
                    st.gpu = false;
                    st.gpu_note = format!("GPU unavailable: {err} — running on CPU");
                }
            }
        }
        let _ = &lock_gen;
        cpu_solve_loop(&solver, &stop, &app, &req);
    });
    session.worker = Some(handle);
    Ok(Json(serde_json::json!({"ok": true})))
}

fn cpu_solve_loop(
    solver: &Arc<Mutex<Solver>>,
    stop: &AtomicBool,
    app: &Arc<AppState>,
    req: &SolveRequest,
) {
    let start = std::time::Instant::now();
    let pot = solver.lock().unwrap().spot.tree.config.starting_pot;
    loop {
        if stop.load(Ordering::Relaxed) {
            solver.lock().unwrap().ensure_symmetric();
            let mut st = app.status.lock().unwrap();
            st.state = "stopped".to_string();
            break;
        }
        let it = {
            let mut s = solver.lock().unwrap();
            s.iterate();
            s.iteration
        };
        let check = it % req.check_every.max(1) == 0 || it >= req.max_iterations;
        if check {
            let e = {
                let s = solver.lock().unwrap();
                s.exploitability()
            };
            let pct = e / pot * 100.0;
            let mut st = app.status.lock().unwrap();
            st.iteration = it;
            st.exploit_chips = e;
            st.exploit_pct = pct;
            st.elapsed_secs = start.elapsed().as_secs_f64();
            st.history.push(HistoryPoint {
                iteration: it,
                exploit_pct: pct,
            });
            if pct <= req.target_exploit_pct || it >= req.max_iterations {
                drop(st);
                solver.lock().unwrap().ensure_symmetric();
                let mut st = app.status.lock().unwrap();
                st.state = "done".to_string();
                break;
            }
        } else {
            let mut st = app.status.lock().unwrap();
            st.iteration = it;
            st.elapsed_secs = start.elapsed().as_secs_f64();
        }
    }
}

/// GPU-backed solve loop: iterations run in VRAM; the CPU solver is refreshed
/// at every exploitability check (and at stop/finish) so queries, saves and
/// locks keep working against near-current data.
#[cfg(feature = "gpu")]
fn gpu_solve_loop(
    solver: &Arc<Mutex<Solver>>,
    stop: &AtomicBool,
    lock_gen: &std::sync::atomic::AtomicU64,
    app: &Arc<AppState>,
    req: &SolveRequest,
) -> Result<(), String> {
    use solver::gpu::GpuSolver;
    let (mut gpu, pot) = {
        let s = solver.lock().unwrap();
        let est = solver::gpu::estimate_vram(&s.spot);
        let cap_mb = gpu_budget().0;
        if est > cap_mb * 1_000_000 {
            return Err(format!(
                "spot needs ~{:.0} MB VRAM (only ~{} MB free)",
                est as f64 / 1e6,
                cap_mb
            ));
        }
        (GpuSolver::new(&s)?, s.spot.tree.config.starting_pot)
    };
    {
        let mut st = app.status.lock().unwrap();
        st.gpu = true;
        st.gpu_note = String::new();
    }
    println!("solving on GPU");
    let start = std::time::Instant::now();
    let mut seen_gen = lock_gen.load(Ordering::Relaxed);
    let mut check_n = 0u32;
    loop {
        if stop.load(Ordering::Relaxed) {
            let mut s = solver.lock().unwrap();
            gpu.sync_to_cpu(&mut s)?;
            s.ensure_symmetric();
            drop(s);
            let mut st = app.status.lock().unwrap();
            st.state = "stopped".to_string();
            return Ok(());
        }
        let g = lock_gen.load(Ordering::Relaxed);
        if g != seen_gen {
            seen_gen = g;
            let s = solver.lock().unwrap();
            gpu.update_locks(&s)?;
        }
        gpu.iterate()?;
        let it = gpu.iteration;
        let check = it % req.check_every.max(1) == 0 || it >= req.max_iterations;
        if check {
            check_n += 1;
            let (e, finished) = {
                let mut s = solver.lock().unwrap();
                // best response runs on the GPU (~50ms); the full arena
                // download is only paid when the solve ends
                let e = gpu.exploitability(&s)?;
                let pct = e / pot * 100.0;
                let finished = pct <= req.target_exploit_pct || it >= req.max_iterations;
                if finished {
                    gpu.sync_to_cpu(&mut s)?;
                    s.ensure_symmetric();
                } else if check_n % 4 == 0 {
                    // keep mid-solve browsing reasonably fresh without paying
                    // the PCIe cost at every check
                    gpu.sync_strategy(&mut s)?;
                }
                (e, finished)
            };
            let pct = e / pot * 100.0;
            let mut st = app.status.lock().unwrap();
            st.iteration = it;
            st.exploit_chips = e;
            st.exploit_pct = pct;
            st.elapsed_secs = start.elapsed().as_secs_f64();
            st.history.push(HistoryPoint {
                iteration: it,
                exploit_pct: pct,
            });
            if finished {
                st.state = "done".to_string();
                return Ok(());
            }
        } else {
            let mut st = app.status.lock().unwrap();
            st.iteration = it;
            st.elapsed_secs = start.elapsed().as_secs_f64();
        }
    }
}

async fn stop_solve(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let st = state.clone();
    tokio::task::spawn_blocking(move || stop_current(&st, false))
        .await
        .ok();
    Json(serde_json::json!({"ok": true}))
}

fn stop_current(state: &Arc<AppState>, drop_session: bool) {
    let mut guard = state.session.lock().unwrap();
    if let Some(session) = guard.as_mut() {
        session.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = session.worker.take() {
            let _ = handle.join();
        }
        if drop_session {
            *guard = None;
            let mut status = state.status.lock().unwrap();
            *status = StatusInfo {
                state: "idle".to_string(),
                ..Default::default()
            };
        }
    }
}

async fn get_status(State(state): State<Arc<AppState>>) -> Json<StatusInfo> {
    Json(state.status.lock().unwrap().clone())
}

#[derive(Deserialize)]
struct NodeRequest {
    path: Vec<PathStep>,
}

async fn get_node(
    State(state): State<Arc<AppState>>,
    Json(req): Json<NodeRequest>,
) -> Result<Json<solver::query::NodeView>, ApiError> {
    let solver = {
        let guard = state.session.lock().unwrap();
        guard
            .as_ref()
            .map(|s| s.solver.clone())
            .ok_or_else(|| bad_request("no spot built yet"))?
    };
    let view = tokio::task::spawn_blocking(move || {
        let mut s = solver.lock().unwrap();
        s.ensure_symmetric();
        s.node_view(&req.path)
    })
    .await
    .map_err(|e| bad_request(e.to_string()))?
    .map_err(bad_request)?;
    Ok(Json(view))
}

// ---------------------------------------------------------------------------
// Preflop solver (multiway, equity-model postflop)
// ---------------------------------------------------------------------------

/// The 169-class pairwise equity table: built once (Monte Carlo, rayon) and
/// cached on disk; ~1 minute cold, instant afterwards.
fn preflop_equity() -> Arc<solver::preflop::equity::EquityTable> {
    static T: std::sync::OnceLock<Arc<solver::preflop::equity::EquityTable>> =
        std::sync::OnceLock::new();
    T.get_or_init(|| {
        let samples = std::env::var("PREFLOP_EQ_SAMPLES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(20_000);
        Arc::new(solver::preflop::equity::EquityTable::load_or_build(
            "cache/preflop_eq169.bin",
            samples,
        ))
    })
    .clone()
}

async fn pf_build(
    State(state): State<Arc<AppState>>,
    Json(cfg): Json<solver::preflop::PreflopConfig>,
) -> Result<Json<serde_json::Value>, ApiError> {
    // stop a running preflop solve before replacing the session
    if let Some(s) = state.preflop.lock().unwrap().as_ref() {
        s.stop.store(true, Ordering::Relaxed);
    }
    let built = tokio::task::spawn_blocking(move || {
        let eq = preflop_equity();
        solver::preflop::PreflopSolver::new(cfg, eq)
    })
    .await
    .map_err(|e| bad_request(e.to_string()))?
    .map_err(bad_request)?;
    let nodes = built.nodes.len();
    let action_nodes = built.nodes.iter().filter(|n| n.kind == 0).count();
    let arena_mb = built.arena_mb();
    let mut status = PreflopStatus::default();
    status.state = "idle".into();
    *state.preflop.lock().unwrap() = Some(PreflopSession {
        solver: Arc::new(Mutex::new(built)),
        stop: Arc::new(AtomicBool::new(false)),
        status: Arc::new(Mutex::new(status)),
    });
    Ok(Json(serde_json::json!({
        "nodes": nodes, "action_nodes": action_nodes, "arena_mb": arena_mb
    })))
}

/// Dry-run tree sizing for the lab's live estimate — no state touched.
async fn pf_estimate(
    Json(cfg): Json<solver::preflop::PreflopConfig>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let est = tokio::task::spawn_blocking(move || solver::preflop::estimate_tree(&cfg))
        .await
        .map_err(|e| bad_request(e.to_string()))?
        .map_err(bad_request)?;
    let arena_mb = est.arena_len as f64 * 8.0 / 1e6;
    let (limit_nodes, limit_mb) =
        (solver::preflop::limit_nodes(), solver::preflop::limit_arena_mb());
    let ok = !est.truncated && est.nodes <= limit_nodes && arena_mb <= limit_mb;
    Ok(Json(serde_json::json!({
        "nodes": est.nodes,
        "action_nodes": est.action_nodes,
        "arena_mb": arena_mb,
        "truncated": est.truncated,
        "ok": ok,
        "limit_nodes": limit_nodes,
        "limit_arena_mb": limit_mb,
    })))
}

#[derive(Deserialize)]
struct PfSolveRequest {
    #[serde(default = "pf_default_iterations")]
    iterations: u32,
    #[serde(default = "pf_default_check")]
    check_every: u32,
    /// Stop when the summed best-response gap (bb) drops below this.
    #[serde(default = "pf_default_target")]
    target_gap: f64,
}
fn pf_default_iterations() -> u32 {
    2000
}
fn pf_default_check() -> u32 {
    50
}
fn pf_default_target() -> f64 {
    0.01
}

fn pf_session(
    state: &AppState,
) -> Result<(Arc<Mutex<solver::preflop::PreflopSolver>>, Arc<AtomicBool>, Arc<Mutex<PreflopStatus>>), ApiError>
{
    state
        .preflop
        .lock()
        .unwrap()
        .as_ref()
        .map(|s| (s.solver.clone(), s.stop.clone(), s.status.clone()))
        .ok_or_else(|| bad_request("no preflop game built yet"))
}

async fn pf_solve(
    State(state): State<Arc<AppState>>,
    Json(req): Json<PfSolveRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let (solver, stop, status) = pf_session(&state)?;
    stop.store(false, Ordering::Relaxed);
    {
        let mut st = status.lock().unwrap();
        st.state = "running".into();
    }
    std::thread::spawn(move || {
        let max = req.iterations.max(1);
        let check = req.check_every.max(1);
        let mut done = 0u32;

        // GPU when built with the feature, enabled, and the game fits the
        // VRAM budget; anything else — including a mid-solve CUDA error —
        // falls back to the CPU + system RAM without losing progress.
        #[cfg(feature = "gpu")]
        let mut gpu: Option<solver::preflop::gpu::PreflopGpu> = if gpu_enabled() {
            let budget = gpu_budget().0;
            let s = solver.lock().unwrap();
            match solver::preflop::gpu::PreflopGpu::new(&s, budget) {
                Ok(g) => {
                    let mut st = status.lock().unwrap();
                    st.gpu = true;
                    st.gpu_note = String::new();
                    println!("preflop solving on GPU");
                    Some(g)
                }
                Err(err) => {
                    println!("preflop gpu unavailable ({err}); solving on CPU");
                    let mut st = status.lock().unwrap();
                    st.gpu = false;
                    st.gpu_note =
                        format!("GPU unavailable: {err} — solving on CPU + system RAM");
                    None
                }
            }
        } else {
            None
        };

        loop {
            if stop.load(Ordering::Relaxed) {
                #[cfg(feature = "gpu")]
                if let Some(g) = gpu.as_ref() {
                    let mut s = solver.lock().unwrap();
                    let _ = g.sync_to_cpu(&mut s);
                }
                status.lock().unwrap().state = "stopped".into();
                return;
            }
            let mut s = solver.lock().unwrap();
            #[cfg(feature = "gpu")]
            {
                let mut failed: Option<String> = None;
                match gpu.as_mut() {
                    Some(g) => {
                        if let Err(err) = g.iterate(&mut s) {
                            failed = Some(err);
                        }
                    }
                    None => s.iterate(),
                }
                if let Some(err) = failed {
                    println!("preflop gpu failed mid-solve ({err}); continuing on CPU");
                    if let Some(g) = gpu.take() {
                        let _ = g.sync_to_cpu(&mut s);
                    }
                    {
                        let mut st = status.lock().unwrap();
                        st.gpu = false;
                        st.gpu_note =
                            format!("GPU failed mid-solve: {err} — continuing on CPU");
                    }
                    s.iterate();
                }
            }
            #[cfg(not(feature = "gpu"))]
            s.iterate();
            done += 1;
            let iteration = s.iteration;
            let checkpoint = done % check == 0 || done >= max;
            if !checkpoint {
                // publish the live iteration every pass so the UI's counter,
                // progress bar and hand grid move continuously; the (costly)
                // best-response gap still runs only at checkpoints
                drop(s);
                let mut st = status.lock().unwrap();
                st.iteration = iteration;
                st.phase = "iterating".into();
                continue;
            }
            {
                // announce the accuracy pass BEFORE it runs: on big trees it
                // holds the solver for a while and the UI should say so
                {
                    let mut st = status.lock().unwrap();
                    st.iteration = iteration;
                    st.phase = "measuring".into();
                }
                #[cfg(feature = "gpu")]
                let (gaps, evs) = match gpu.as_mut() {
                    Some(g) => match g.gaps_and_evs() {
                        Ok(ge) => {
                            // keep browse/export in sync with the device
                            let _ = g.sync_to_cpu(&mut s);
                            ge
                        }
                        Err(err) => {
                            println!("preflop gpu checkpoint failed ({err}); on CPU");
                            if let Some(g) = gpu.take() {
                                let _ = g.sync_to_cpu(&mut s);
                            }
                            let mut st = status.lock().unwrap();
                            st.gpu = false;
                            st.gpu_note = format!("GPU failed: {err} — continuing on CPU");
                            drop(st);
                            s.gaps_and_evs()
                        }
                    },
                    None => s.gaps_and_evs(),
                };
                #[cfg(not(feature = "gpu"))]
                let (gaps, evs) = s.gaps_and_evs();
                drop(s);
                let total: f64 = gaps.iter().sum();
                let mut st = status.lock().unwrap();
                st.iteration = iteration;
                st.phase = "iterating".into();
                st.gaps = gaps;
                st.gap_total = total;
                st.evs = evs;
                if total < req.target_gap || done >= max {
                    st.state = "done".into();
                    return;
                }
            }
        }
    });
    Ok(Json(serde_json::json!({"ok": true})))
}

async fn pf_stop(State(state): State<Arc<AppState>>) -> Result<Json<serde_json::Value>, ApiError> {
    let (_, stop, _) = pf_session(&state)?;
    stop.store(true, Ordering::Relaxed);
    Ok(Json(serde_json::json!({"ok": true})))
}

async fn pf_status(State(state): State<Arc<AppState>>) -> Json<PreflopStatus> {
    let st = state
        .preflop
        .lock()
        .unwrap()
        .as_ref()
        .map(|s| s.status.lock().unwrap().clone())
        .unwrap_or_default();
    Json(st)
}

#[derive(Deserialize)]
struct PfPathRequest {
    path: Vec<usize>,
}

// ---- player profiles ----

#[derive(Deserialize)]
struct PfSeatModel {
    #[serde(default)]
    frozen: bool,
    #[serde(default)]
    profile: Option<solver::preflop::SeatProfile>,
}

#[derive(Deserialize)]
struct PfTableRequest {
    seats: Vec<PfSeatModel>,
}

async fn pf_table(
    State(state): State<Arc<AppState>>,
    Json(req): Json<PfTableRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let (solver, _, _) = pf_session(&state)?;
    let overrides = tokio::task::spawn_blocking(move || {
        let mut s = solver.lock().unwrap();
        let frozen = req.seats.iter().map(|x| x.frozen).collect();
        let profiles = req.seats.into_iter().map(|x| x.profile).collect();
        s.set_table(frozen, profiles)?;
        Ok::<bool, String>(s.has_overrides())
    })
    .await
    .map_err(|e| bad_request(e.to_string()))?
    .map_err(bad_request)?;
    Ok(Json(serde_json::json!({"ok": true, "overrides": overrides})))
}

#[derive(Deserialize)]
struct PfGenerateRequest {
    seat: usize,
    stats: solver::preflop::HudStats,
    #[serde(default)]
    name: String,
}

async fn pf_generate(
    State(state): State<Arc<AppState>>,
    Json(req): Json<PfGenerateRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let (solver, _, _) = pf_session(&state)?;
    let out = tokio::task::spawn_blocking(move || {
        let s = solver.lock().unwrap();
        let name = if req.name.is_empty() { "custom" } else { &req.name };
        s.generate_profile(req.seat, &req.stats, name)
    })
    .await
    .map_err(|e| bad_request(e.to_string()))?
    .map_err(bad_request)?;
    Ok(Json(serde_json::json!({"profile": out.0, "implied": out.1})))
}

#[derive(Deserialize)]
struct ProfileLocksRequest {
    /// Villain seat in the postflop spot: 0 = OOP, 1 = IP.
    player: usize,
    stats: solver::query::PostflopStats,
    /// Who arrives at the flop with the initiative (last preflop raiser).
    #[serde(default)]
    aggressor: Option<usize>,
}

/// Compile a postflop stat profile into node locks across the villain's
/// whole tree. Returns {locked, rows: [{label, target, achieved}]}.
async fn profile_locks(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ProfileLocksRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let (solver, lock_gen) = {
        let guard = state.session.lock().unwrap();
        guard
            .as_ref()
            .map(|s| (s.solver.clone(), s.lock_gen.clone()))
            .ok_or_else(|| bad_request("no spot built yet"))?
    };
    let summary = tokio::task::spawn_blocking(move || {
        let mut s = solver.lock().unwrap();
        s.lock_profile(req.player, &req.stats, req.aggressor)
    })
    .await
    .map_err(|e| bad_request(e.to_string()))?
    .map_err(bad_request)?;
    lock_gen.fetch_add(1, Ordering::Relaxed);
    Ok(Json(serde_json::json!(summary)))
}

async fn profile_locks_clear(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let (solver, lock_gen) = {
        let guard = state.session.lock().unwrap();
        guard
            .as_ref()
            .map(|s| (s.solver.clone(), s.lock_gen.clone()))
            .ok_or_else(|| bad_request("no spot built yet"))?
    };
    let n = tokio::task::spawn_blocking(move || {
        let mut s = solver.lock().unwrap();
        s.clear_profile_locks()
    })
    .await
    .map_err(|e| bad_request(e.to_string()))?;
    lock_gen.fetch_add(1, Ordering::Relaxed);
    Ok(Json(serde_json::json!({"cleared": n})))
}

fn pf_game_path(name: &str) -> Result<std::path::PathBuf, String> {
    let clean: String = name
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == ' ' || *c == '-' || *c == '_')
        .collect();
    if clean.trim().is_empty() {
        return Err("give the save a name".into());
    }
    Ok(std::path::PathBuf::from("saves/preflop").join(format!("{}.gtop", clean.trim())))
}

#[derive(Deserialize)]
struct PfGameName {
    name: String,
}

async fn pf_save_game(
    State(state): State<Arc<AppState>>,
    Json(req): Json<PfGameName>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let (solver, _, status) = pf_session(&state)?;
    if status.lock().unwrap().state == "running" {
        return Err(bad_request("stop the solve first, then save"));
    }
    let path = pf_game_path(&req.name).map_err(bad_request)?;
    std::fs::create_dir_all("saves/preflop").map_err(|e| bad_request(e.to_string()))?;
    let iteration = tokio::task::spawn_blocking(move || {
        let s = solver.lock().unwrap();
        s.save_game(path.to_str().unwrap())?;
        Ok::<u32, String>(s.iteration)
    })
    .await
    .map_err(|e| bad_request(e.to_string()))?
    .map_err(bad_request)?;
    Ok(Json(serde_json::json!({ "ok": true, "iteration": iteration })))
}

async fn pf_load_game(
    State(state): State<Arc<AppState>>,
    Json(req): Json<PfGameName>,
) -> Result<Json<serde_json::Value>, ApiError> {
    // stop a running preflop solve before replacing the session
    if let Some(s) = state.preflop.lock().unwrap().as_ref() {
        s.stop.store(true, Ordering::Relaxed);
    }
    let path = pf_game_path(&req.name).map_err(bad_request)?;
    let loaded = tokio::task::spawn_blocking(move || {
        solver::preflop::PreflopSolver::load_game(path.to_str().unwrap(), preflop_equity())
    })
    .await
    .map_err(|e| bad_request(e.to_string()))?
    .map_err(bad_request)?;
    let seats: Vec<serde_json::Value> = (0..loaded.cfg.positions.len())
        .map(|i| {
            serde_json::json!({
                "frozen": loaded.seat_frozen[i],
                "profile": loaded.seat_profiles[i],
            })
        })
        .collect();
    let out = serde_json::json!({
        "config": loaded.cfg,
        "nodes": loaded.nodes.len(),
        "action_nodes": loaded.nodes.iter().filter(|n| n.kind == 0).count(),
        "arena_mb": loaded.arena_mb(),
        "iteration": loaded.iteration,
        "seats": seats,
    });
    let status = PreflopStatus {
        state: "stopped".into(),
        iteration: loaded.iteration,
        ..Default::default()
    };
    *state.preflop.lock().unwrap() = Some(PreflopSession {
        solver: Arc::new(Mutex::new(loaded)),
        stop: Arc::new(AtomicBool::new(false)),
        status: Arc::new(Mutex::new(status)),
    });
    Ok(Json(out))
}

async fn pf_list_games() -> Json<serde_json::Value> {
    let mut entries: Vec<(String, std::time::SystemTime)> = Vec::new();
    if let Ok(rd) = std::fs::read_dir("saves/preflop") {
        for e in rd.flatten() {
            let p = e.path();
            if p.extension().and_then(|x| x.to_str()) == Some("gtop") {
                if let Some(stem) = p.file_stem().and_then(|x| x.to_str()) {
                    let t = e.metadata().and_then(|m| m.modified()).unwrap_or(std::time::UNIX_EPOCH);
                    entries.push((stem.to_string(), t));
                }
            }
        }
    }
    entries.sort_by(|a, b| b.1.cmp(&a.1)); // newest first
    Json(serde_json::json!(entries.into_iter().map(|(n, _)| n).collect::<Vec<_>>()))
}

// ---------------------------------------------------------------------------
// Flop reports: solve one spot config across a canonical flop subset in a
// background thread, extracting per-flop aggregates (and optionally locking
// a villain to his postflop profile before measuring).
// ---------------------------------------------------------------------------

#[derive(Clone, Serialize, Deserialize)]
struct ReportVillain {
    /// 0 = OOP, 1 = IP.
    player: usize,
    name: String,
    stats: solver::query::PostflopStats,
    #[serde(default)]
    aggressor: Option<usize>,
}

#[derive(Deserialize)]
struct ReportRequest {
    name: String,
    spot: SpotRequest,
    #[serde(default = "report_dflt_flops")]
    flops: usize,
    #[serde(default = "report_dflt_iters")]
    max_iterations: u32,
    #[serde(default = "report_dflt_target")]
    target: f64,
    #[serde(default)]
    villain: Option<ReportVillain>,
}
fn report_dflt_flops() -> usize {
    95
}
fn report_dflt_iters() -> u32 {
    600
}
fn report_dflt_target() -> f64 {
    0.35
}

fn report_path(name: &str) -> Result<std::path::PathBuf, String> {
    let clean: String = name
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == ' ' || *c == '-' || *c == '_')
        .collect();
    if clean.trim().is_empty() {
        return Err("give the report a name".into());
    }
    Ok(std::path::PathBuf::from("saves/reports").join(format!("{}.json", clean.trim())))
}

/// Reach-weighted per-player aggregates + action frequencies at a node.
fn report_node_stats(
    view: &solver::query::NodeView,
    pot: f64,
) -> (Vec<serde_json::Value>, Option<serde_json::Value>) {
    let mut players = Vec::new();
    for p in 0..2 {
        let (mut wev, mut weq, mut wt) = (0f64, 0f64, 0f64);
        for h in &view.players[p].hands {
            if let (Some(eq), Some(ev)) = (h.eq, h.ev) {
                wev += ev as f64 * h.reach as f64;
                weq += eq as f64 * h.reach as f64;
                wt += h.reach as f64;
            }
        }
        let (ev, eq) = if wt > 1e-12 { (wev / wt, weq / wt) } else { (0.0, 0.0) };
        let eqr = if eq > 0.02 { ev / (pot * eq) } else { 0.0 };
        players.push(serde_json::json!({"ev": ev, "eq": eq, "eqr": eqr}));
    }
    let strat = view.player.map(|actor| {
        let actor = actor as usize;
        let na = view.actions.len();
        let (mut sums, mut total) = (vec![0f64; na], 0f64);
        for h in &view.players[actor].hands {
            if let Some(st) = &h.strategy {
                total += h.reach as f64;
                for a in 0..na {
                    sums[a] += st[a] as f64 * h.reach as f64;
                }
            }
        }
        let freqs: Vec<f64> =
            sums.iter().map(|s| if total > 1e-12 { s / total } else { 0.0 }).collect();
        serde_json::json!({
            "actor": actor,
            "actions": view.actions.iter().map(|a| a.label.clone()).collect::<Vec<_>>(),
            "kinds": view.actions.iter().map(|a| a.kind.clone()).collect::<Vec<_>>(),
            "freqs": freqs,
        })
    });
    (players, strat)
}

async fn report_run(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ReportRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    {
        let r = state.report.lock().unwrap();
        if r.running {
            return Err((StatusCode::CONFLICT, "a report is already running".into()));
        }
    }
    let path = report_path(&req.name).map_err(bad_request)?;
    std::fs::create_dir_all("saves/reports").map_err(|e| bad_request(e.to_string()))?;
    // validate the config once before going background
    let mut probe = req.spot.clone();
    probe.board = "AhKs2d".into();
    probe.to_spot_config().map_err(bad_request)?;

    let flops = solver::cards::canonical_flops_subset(req.flops);
    {
        let mut r = state.report.lock().unwrap();
        *r = ReportStatus {
            running: true,
            name: req.name.clone(),
            total: flops.len(),
            ..Default::default()
        };
    }
    state.report_stop.store(false, Ordering::Relaxed);
    let app = state.clone();
    std::thread::spawn(move || {
        let t0 = std::time::Instant::now();
        let mut rows: Vec<serde_json::Value> = Vec::new();
        let mut err = String::new();
        for (i, (board, weight)) in flops.iter().enumerate() {
            if app.report_stop.load(Ordering::Relaxed) {
                break;
            }
            {
                let mut r = app.report.lock().unwrap();
                r.done = i;
                r.board = board.clone();
                r.seconds = t0.elapsed().as_secs_f64();
            }
            let mut sr = req.spot.clone();
            sr.board = board.clone();
            let cfg = match sr.to_spot_config() {
                Ok(c) => c,
                Err(e) => {
                    err = e;
                    break;
                }
            };
            let spot = match Spot::new(cfg) {
                Ok(s) => s,
                Err(e) => {
                    err = e;
                    break;
                }
            };
            let mut solver = Solver::with_storage(Arc::new(spot), Storage::Compressed);
            let bt0 = std::time::Instant::now();
            let (iters, pct) =
                report_solve(&mut solver, req.max_iterations, req.target, &app.report_stop);
            let mut lock_summary = serde_json::Value::Null;
            if let Some(v) = &req.villain {
                match solver.lock_profile(v.player, &v.stats, v.aggressor) {
                    Ok(sm) => {
                        lock_summary = serde_json::json!({"locked": sm.locked});
                        // hero re-adapts against the locked villain
                        let _ = report_solve(
                            &mut solver,
                            req.max_iterations / 2,
                            req.target,
                            &app.report_stop,
                        );
                    }
                    Err(e) => {
                        err = format!("villain lock failed on {board}: {e}");
                        break;
                    }
                }
            }
            solver.ensure_symmetric();
            let view = match solver.node_view(&[]) {
                Ok(v) => v,
                Err(e) => {
                    err = e;
                    break;
                }
            };
            let pot = req.spot.starting_pot;
            let (players, root_strat) = report_node_stats(&view, pot);
            // IP's response after a root check, when the root has one
            let mut vs_check = serde_json::Value::Null;
            if let Some(ci) = view.actions.iter().position(|a| a.kind == "check") {
                if let Ok(v2) = solver.node_view(&[solver::query::PathStep::Action { index: ci }])
                {
                    let (_, st2) = report_node_stats(&v2, pot);
                    if let Some(st2) = st2 {
                        vs_check = st2;
                    }
                }
            }
            rows.push(serde_json::json!({
                "board": board, "weight": weight, "iterations": iters,
                "exploit_pct": pct, "seconds": bt0.elapsed().as_secs_f64(),
                "players": players, "root": root_strat, "vs_check": vs_check,
                "villain_lock": lock_summary,
            }));
            if rows.len() % 8 == 0 {
                let _ = write_report(&path, &req, &rows, false);
            }
        }
        let done = rows.len();
        if let Err(e) = write_report(&path, &req, &rows, err.is_empty()) {
            if err.is_empty() {
                err = e;
            }
        }
        let mut r = app.report.lock().unwrap();
        r.running = false;
        r.done = done;
        r.seconds = t0.elapsed().as_secs_f64();
        r.error = err;
    });
    Ok(Json(serde_json::json!({"ok": true})))
}

fn report_solve(
    solver: &mut Solver,
    max_iterations: u32,
    target: f64,
    stop: &AtomicBool,
) -> (u32, f64) {
    let pot = solver.spot.tree.config.starting_pot;
    let base = solver.iteration;
    // fresh solves take the GPU when built with it (villain re-adapt solves
    // continue on CPU: they start from synced state and are short)
    #[cfg(feature = "gpu")]
    if gpu_enabled() && base == 0 {
        if let Ok(mut gpu) = solver::gpu::GpuSolver::new(solver) {
            loop {
                if stop.load(Ordering::Relaxed) {
                    let _ = gpu.sync_to_cpu(solver);
                    return (gpu.iteration, -1.0);
                }
                if gpu.iterate().is_err() {
                    break; // fall through to CPU
                }
                let it = gpu.iteration;
                if it % 20 == 0 || it >= max_iterations {
                    let e = match gpu.exploitability(solver) {
                        Ok(e) => e,
                        Err(_) => break,
                    };
                    let pct = e / pot * 100.0;
                    if pct <= target || it >= max_iterations {
                        if gpu.sync_to_cpu(solver).is_ok() {
                            return (it, pct);
                        }
                        break;
                    }
                }
            }
        }
    }
    loop {
        if stop.load(Ordering::Relaxed) {
            return (solver.iteration, -1.0);
        }
        solver.iterate();
        let it = solver.iteration - base;
        if it % 20 == 0 || it >= max_iterations {
            let pct = solver.exploitability() / pot * 100.0;
            if pct <= target || it >= max_iterations {
                return (solver.iteration, pct);
            }
        }
    }
}

fn write_report(
    path: &std::path::Path,
    req: &ReportRequest,
    rows: &[serde_json::Value],
    complete: bool,
) -> Result<(), String> {
    let out = serde_json::json!({
        "name": req.name,
        "created": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        "spot": req.spot,
        "villain": req.villain,
        "target_pct": req.target,
        "complete": complete,
        "flops": rows,
    });
    std::fs::write(path, serde_json::to_string(&out).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())
}

async fn report_status(State(state): State<Arc<AppState>>) -> Json<ReportStatus> {
    Json(state.report.lock().unwrap().clone())
}

async fn report_stop_run(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    state.report_stop.store(true, Ordering::Relaxed);
    Json(serde_json::json!({"ok": true}))
}

async fn report_list() -> Json<serde_json::Value> {
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir("saves/reports") {
        for e in rd.flatten() {
            let p = e.path();
            if p.extension().and_then(|x| x.to_str()) != Some("json") {
                continue;
            }
            if let Ok(text) = std::fs::read_to_string(&p) {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) {
                    out.push(serde_json::json!({
                        "name": v.get("name"),
                        "created": v.get("created"),
                        "n_flops": v.get("flops").and_then(|f| f.as_array()).map(|a| a.len()),
                        "complete": v.get("complete"),
                        "villain": v.get("villain").and_then(|x| x.get("name")),
                        "board_sample": v.get("flops").and_then(|f| f.get(0)).and_then(|r| r.get("board")),
                    }));
                }
            }
        }
    }
    out.sort_by_key(|v| -(v.get("created").and_then(|c| c.as_i64()).unwrap_or(0)));
    Json(serde_json::json!(out))
}

#[derive(Deserialize)]
struct ReportName {
    name: String,
}

async fn report_get(Json(req): Json<ReportName>) -> Result<Json<serde_json::Value>, ApiError> {
    let path = report_path(&req.name).map_err(bad_request)?;
    let text = std::fs::read_to_string(&path).map_err(|e| bad_request(e.to_string()))?;
    Ok(Json(serde_json::from_str(&text).map_err(|e| bad_request(e.to_string()))?))
}

async fn report_delete(Json(req): Json<ReportName>) -> Result<Json<serde_json::Value>, ApiError> {
    let path = report_path(&req.name).map_err(bad_request)?;
    std::fs::remove_file(&path).map_err(|e| bad_request(e.to_string()))?;
    Ok(Json(serde_json::json!({"ok": true})))
}

async fn pf_archetypes() -> Json<serde_json::Value> {
    let list: Vec<serde_json::Value> = solver::preflop::archetypes()
        .into_iter()
        .map(|(n, s)| {
            let pf = solver::preflop::archetype_postflop(n);
            serde_json::json!({"name": n, "stats": s, "postflop": pf})
        })
        .collect();
    Json(serde_json::json!(list))
}

#[derive(Deserialize)]
struct PfHeroRequest {
    seat: Option<usize>,
}

async fn pf_hero(
    State(state): State<Arc<AppState>>,
    Json(req): Json<PfHeroRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let (solver, _, _) = pf_session(&state)?;
    tokio::task::spawn_blocking(move || {
        let mut s = solver.lock().unwrap();
        s.set_hero(req.seat)
    })
    .await
    .map_err(|e| bad_request(e.to_string()))?
    .map_err(bad_request)?;
    Ok(Json(serde_json::json!({"ok": true})))
}

fn profile_path(name: &str) -> Result<std::path::PathBuf, String> {
    let clean: String = name
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == ' ' || *c == '-' || *c == '_')
        .collect();
    if clean.trim().is_empty() {
        return Err("profile needs a name".into());
    }
    Ok(std::path::PathBuf::from("saves/profiles").join(format!("{}.json", clean.trim())))
}

async fn pf_profiles_list() -> Json<serde_json::Value> {
    let mut names: Vec<String> = std::fs::read_dir("saves/profiles")
        .map(|rd| {
            rd.flatten()
                .filter_map(|e| {
                    e.path()
                        .file_stem()
                        .map(|s| s.to_string_lossy().to_string())
                })
                .collect()
        })
        .unwrap_or_default();
    names.sort();
    Json(serde_json::json!(names))
}

#[derive(Deserialize)]
struct PfProfileSave {
    name: String,
    profile: solver::preflop::SeatProfile,
}

async fn pf_profile_save(
    Json(req): Json<PfProfileSave>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let path = profile_path(&req.name).map_err(bad_request)?;
    std::fs::create_dir_all("saves/profiles").map_err(|e| bad_request(e.to_string()))?;
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(&req.profile).map_err(|e| bad_request(e.to_string()))?,
    )
    .map_err(|e| bad_request(e.to_string()))?;
    Ok(Json(serde_json::json!({"ok": true})))
}

#[derive(Deserialize)]
struct PfProfileGet {
    name: String,
}

async fn pf_profile_get(
    Json(req): Json<PfProfileGet>,
) -> Result<Json<solver::preflop::SeatProfile>, ApiError> {
    let path = profile_path(&req.name).map_err(bad_request)?;
    let bytes = std::fs::read(&path).map_err(|e| bad_request(e.to_string()))?;
    let p: solver::preflop::SeatProfile =
        serde_json::from_slice(&bytes).map_err(|e| bad_request(e.to_string()))?;
    Ok(Json(p))
}

#[derive(Deserialize)]
struct PfPointLockRequest {
    path: Vec<usize>,
    #[serde(default)]
    policy: Option<solver::preflop::BucketPolicy>,
}

async fn pf_lock(
    State(state): State<Arc<AppState>>,
    Json(req): Json<PfPointLockRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let (solver, _, _) = pf_session(&state)?;
    tokio::task::spawn_blocking(move || {
        let mut s = solver.lock().unwrap();
        s.lock_point(&req.path, req.policy)
    })
    .await
    .map_err(|e| bad_request(e.to_string()))?
    .map_err(bad_request)?;
    Ok(Json(serde_json::json!({"ok": true})))
}

async fn pf_unlock(
    State(state): State<Arc<AppState>>,
    Json(req): Json<PfPathRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let (solver, _, _) = pf_session(&state)?;
    let removed = tokio::task::spawn_blocking(move || {
        let mut s = solver.lock().unwrap();
        s.unlock_point(&req.path)
    })
    .await
    .map_err(|e| bad_request(e.to_string()))?
    .map_err(bad_request)?;
    Ok(Json(serde_json::json!({"ok": true, "removed": removed})))
}

async fn pf_node(
    State(state): State<Arc<AppState>>,
    Json(req): Json<PfPathRequest>,
) -> Result<Json<solver::preflop::PreflopNodeView>, ApiError> {
    let (solver, _, _) = pf_session(&state)?;
    let view = tokio::task::spawn_blocking(move || {
        let s = solver.lock().unwrap();
        s.node_view(&req.path)
    })
    .await
    .map_err(|e| bad_request(e.to_string()))?
    .map_err(bad_request)?;
    Ok(Json(view))
}

async fn pf_export(
    State(state): State<Arc<AppState>>,
    Json(req): Json<PfPathRequest>,
) -> Result<Json<solver::preflop::PreflopExport>, ApiError> {
    let (solver, _, _) = pf_session(&state)?;
    let out = tokio::task::spawn_blocking(move || {
        let s = solver.lock().unwrap();
        s.export_spot(&req.path)
    })
    .await
    .map_err(|e| bad_request(e.to_string()))?
    .map_err(bad_request)?;
    Ok(Json(out))
}

#[derive(Deserialize)]
struct ExploitRequest {
    path: Vec<PathStep>,
    /// 0 = OOP, 1 = IP: the player whose best response to compute.
    exploiter: u8,
}

async fn exploit_view(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ExploitRequest>,
) -> Result<Json<solver::query::ExploitView>, ApiError> {
    let solver = {
        let guard = state.session.lock().unwrap();
        guard
            .as_ref()
            .map(|s| s.solver.clone())
            .ok_or_else(|| bad_request("no spot built yet"))?
    };
    let view = tokio::task::spawn_blocking(move || {
        let s = solver.lock().unwrap();
        s.exploit_view(&req.path, req.exploiter as usize)
    })
    .await
    .map_err(|e| bad_request(e.to_string()))?
    .map_err(bad_request)?;
    Ok(Json(view))
}

#[derive(Deserialize)]
struct LockRequest {
    path: Vec<PathStep>,
    /// How to build the lock: freeze / range frequencies / per-hand edits.
    mode: solver::query::LockMode,
    #[serde(default)]
    label: String,
}

async fn lock_node(
    State(state): State<Arc<AppState>>,
    Json(req): Json<LockRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let (solver, lock_gen) = {
        let guard = state.session.lock().unwrap();
        guard
            .as_ref()
            .map(|s| (s.solver.clone(), s.lock_gen.clone()))
            .ok_or_else(|| bad_request("no spot built yet"))?
    };
    tokio::task::spawn_blocking(move || {
        let mut s = solver.lock().unwrap();
        s.lock_node(&req.path, req.mode, req.label)
    })
    .await
    .map_err(|e| bad_request(e.to_string()))?
    .map_err(bad_request)?;
    lock_gen.fetch_add(1, Ordering::Relaxed);
    Ok(Json(serde_json::json!({"ok": true})))
}

async fn unlock_node(
    State(state): State<Arc<AppState>>,
    Json(req): Json<NodeRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let (solver, lock_gen) = {
        let guard = state.session.lock().unwrap();
        guard
            .as_ref()
            .map(|s| (s.solver.clone(), s.lock_gen.clone()))
            .ok_or_else(|| bad_request("no spot built yet"))?
    };
    let removed = tokio::task::spawn_blocking(move || {
        let mut s = solver.lock().unwrap();
        s.unlock_node(&req.path)
    })
    .await
    .map_err(|e| bad_request(e.to_string()))?
    .map_err(bad_request)?;
    lock_gen.fetch_add(1, Ordering::Relaxed);
    Ok(Json(serde_json::json!({"ok": true, "removed": removed})))
}

async fn list_locks(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<String>>, ApiError> {
    let solver = {
        let guard = state.session.lock().unwrap();
        guard
            .as_ref()
            .map(|s| s.solver.clone())
            .ok_or_else(|| bad_request("no spot built yet"))?
    };
    let locks = tokio::task::spawn_blocking(move || {
        let s = solver.lock().unwrap();
        s.list_locks()
    })
    .await
    .map_err(|e| bad_request(e.to_string()))?;
    Ok(Json(locks))
}

async fn runouts(
    State(state): State<Arc<AppState>>,
    Json(req): Json<NodeRequest>,
) -> Result<Json<solver::query::RunoutsReport>, ApiError> {
    let solver = {
        let guard = state.session.lock().unwrap();
        guard
            .as_ref()
            .map(|s| s.solver.clone())
            .ok_or_else(|| bad_request("no spot built yet"))?
    };
    let report = tokio::task::spawn_blocking(move || {
        let mut s = solver.lock().unwrap();
        s.ensure_symmetric();
        s.runouts(&req.path)
    })
    .await
    .map_err(|e| bad_request(e.to_string()))?
    .map_err(bad_request)?;
    Ok(Json(report))
}

#[derive(Deserialize)]
struct ParseRangeRequest {
    text: String,
}

#[derive(Serialize)]
struct ParseRangeResponse {
    weights: Vec<f32>,
    combos: f32,
    compact: String,
}

async fn parse_range(Json(req): Json<ParseRangeRequest>) -> Result<Json<ParseRangeResponse>, ApiError> {
    let range = Range::parse(&req.text).map_err(bad_request)?;
    Ok(Json(ParseRangeResponse {
        combos: range.num_combos(),
        compact: range.to_string_compact(),
        weights: range.weights,
    }))
}

// ---------------------------------------------------------------------------
// Save / load
// ---------------------------------------------------------------------------

fn saves_dir() -> std::path::PathBuf {
    let dir = std::path::PathBuf::from("saves");
    std::fs::create_dir_all(&dir).ok();
    dir
}

fn sanitize_name(name: &str) -> Result<String, ApiError> {
    let clean: String = name
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '_' || *c == ' ' || *c == '.')
        .collect();
    let clean = clean.trim().to_string();
    if clean.is_empty() {
        return Err(bad_request("invalid save name"));
    }
    Ok(clean)
}

#[derive(Deserialize)]
struct SaveRequest {
    name: String,
}

async fn save_solve(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SaveRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    // While a GPU solve is running, the CPU-side regret arena can lag the
    // strategy arena (strategy-only syncs); a save now would checkpoint a
    // mismatched state. Stop first to force a full sync.
    if state.status.lock().unwrap().state == "running" {
        return Err((
            StatusCode::CONFLICT,
            "stop the solve before saving (mid-solve saves can checkpoint stale data)".to_string(),
        ));
    }
    let name = sanitize_name(&req.name)?;
    let solver = {
        let guard = state.session.lock().unwrap();
        guard
            .as_ref()
            .map(|s| s.solver.clone())
            .ok_or_else(|| bad_request("no spot built yet"))?
    };
    let path = saves_dir().join(format!("{name}.gto"));
    let path_str = path.to_str().unwrap().to_string();
    tokio::task::spawn_blocking(move || {
        let mut s = solver.lock().unwrap();
        s.ensure_symmetric();
        s.save(&path_str)
    })
    .await
    .map_err(|e| bad_request(e.to_string()))?
    .map_err(bad_request)?;
    Ok(Json(serde_json::json!({"ok": true})))
}

async fn load_solve(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SaveRequest>,
) -> Result<Json<TreeInfo>, ApiError> {
    let name = sanitize_name(&req.name)?;
    let st = state.clone();
    tokio::task::spawn_blocking(move || stop_current(&st, true))
        .await
        .map_err(|e| bad_request(e.to_string()))?;
    let path = saves_dir().join(format!("{name}.gto"));
    let path_str = path
        .to_str()
        .ok_or_else(|| bad_request("bad path"))?
        .to_string();
    let solver = tokio::task::spawn_blocking(move || {
        Solver::load_with_storage(&path_str, storage_from_env())
    })
    .await
    .map_err(|e| bad_request(e.to_string()))?
    .map_err(bad_request)?;

    let spot = &solver.spot;
    let info = tree_info(spot, solver.arena_bytes() as f64 / 1e6);
    let iteration = solver.iteration;
    let spot_request = spot_request_from_config(&spot.config);
    *state.session.lock().unwrap() = Some(Session {
        solver: Arc::new(Mutex::new(solver)),
        stop: Arc::new(AtomicBool::new(false)),
        worker: None,
        lock_gen: Arc::new(std::sync::atomic::AtomicU64::new(0)),
    });
    let mut status = state.status.lock().unwrap();
    *status = StatusInfo {
        state: "done".to_string(),
        iteration,
        tree: Some(info.clone()),
        spot_request,
        ..Default::default()
    };
    Ok(Json(info))
}

fn sizes_to_string(sizes: &[solver::tree::BetSize]) -> String {
    sizes
        .iter()
        .map(|s| match s {
            solver::tree::BetSize::PotPct(p) => format!("{p}"),
            solver::tree::BetSize::PrevMult(m) => format!("{m}x"),
            solver::tree::BetSize::AllIn => "a".to_string(),
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn spot_request_from_config(config: &SpotConfig) -> Option<SpotRequest> {
    let t = &config.tree;
    let conv = |s: &[StreetSizing; 3]| -> Vec<SizesRequest> {
        s.iter()
            .map(|x| SizesRequest {
                bet: sizes_to_string(&x.bet),
                raise: sizes_to_string(&x.raise),
                donk: sizes_to_string(&x.donk),
            })
            .collect()
    };
    Some(SpotRequest {
        board: config.board.clone(),
        range_oop: config.range_oop.clone(),
        range_ip: config.range_ip.clone(),
        starting_pot: t.starting_pot,
        effective_stack: t.effective_stack,
        rake_pct: t.rake_pct * 100.0,
        rake_cap: t.rake_cap,
        allin_threshold: t.allin_threshold * 100.0,
        add_allin: t.add_allin,
        max_raises: t.max_raises,
        oop: conv(&t.oop),
        ip: conv(&t.ip),
    })
}

async fn list_saves() -> Json<Vec<String>> {
    let mut names = Vec::new();
    if let Ok(entries) = std::fs::read_dir(saves_dir()) {
        for e in entries.flatten() {
            if let Some(name) = e.file_name().to_str() {
                if let Some(stem) = name.strip_suffix(".gto") {
                    names.push(stem.to_string());
                }
            }
        }
    }
    names.sort();
    Json(names)
}

async fn get_presets() -> Json<serde_json::Value> {
    Json(serde_json::json!([
        {
            "name": "BTN open (~44%)",
            "range": "22+,A2s+,K5s+,Q7s+,J7s+,T7s+,96s+,86s+,75s+,64s+,54s,43s,A2o+,K9o+,Q9o+,J9o+,T8o+,98o,87o"
        },
        {
            "name": "BB defend vs BTN open",
            "range": "55-22,A8s-A2s,K9s-K2s,Q9s-Q4s,J9s-J6s,T9s-T6s,98s-95s,87s-84s,76s-74s,65s-63s,54s-52s,43s,42s,32s,AJo-A2o,KTo-K8o,QTo-Q8o,JTo-J8o,T9o-T8o,98o,87o,76o,65o"
        },
        {
            "name": "CO open (~30%)",
            "range": "22+,A2s+,K8s+,Q9s+,J9s+,T8s+,97s+,86s+,76s,65s,54s,A8o+,A5o,KTo+,QTo+,JTo,T9o"
        },
        {
            "name": "UTG open (~18%)",
            "range": "44+,A5s-A2s,ATs+,KTs+,QTs+,JTs,T9s,98s,87s,76s,AJo+,KQo"
        },
        {
            "name": "3-bettor (BB vs BTN 3bet)",
            "range": "99+,AJs+,KQs,A5s-A4s,KJs,QJs,JTs,T9s,AQo+,76s,65s"
        },
        {
            "name": "BTN call vs BB 3bet",
            "range": "JJ-22,AQs-A9s,A5s-A4s,KTs+,QTs+,J9s+,T8s+,97s+,87s,76s,65s,54s,AQo-AJo,KQo"
        },
        {
            "name": "Polarized river example",
            "range": "AA,KK,A5s-A2s"
        },
        {
            "name": "Condensed river example",
            "range": "QQ-88,AJs-A9s,KQs,KJs"
        }
    ]))
}

// ---------------------------------------------------------------------------

fn init_rayon() {
    let threads = std::env::var("SOLVER_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or_else(|| {
            // SMT hurts this memory-bound workload; default to physical cores.
            (std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(8)
                / 2)
            .max(1)
        });
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .ok();
    println!("solver threads: {threads}");
}

#[tokio::main]
async fn main() {
    init_rayon();
    let state = Arc::new(AppState {
        session: Mutex::new(None),
        status: Mutex::new(StatusInfo {
            state: "idle".to_string(),
            ..Default::default()
        }),
        preflop: Mutex::new(None),
        report: Mutex::new(ReportStatus::default()),
        report_stop: Arc::new(AtomicBool::new(false)),
    });

    let serve_dir = tower_http::services::ServeDir::new("web")
        .append_index_html_on_directories(true);

    let app = Router::new()
        .route("/api/spot", post(build_spot))
        .route("/api/solve", post(start_solve))
        .route("/api/stop", post(stop_solve))
        .route("/api/status", get(get_status))
        .route("/api/node", post(get_node))
        .route("/api/exploit", post(exploit_view))
        .route("/api/lock", post(lock_node))
        .route("/api/profile_locks", post(profile_locks).delete(profile_locks_clear))
        .route("/api/reports/run", post(report_run))
        .route("/api/reports/status", get(report_status))
        .route("/api/reports/stop", post(report_stop_run))
        .route("/api/reports", get(report_list))
        .route("/api/reports/get", post(report_get))
        .route("/api/reports/delete", post(report_delete))
        .route("/api/unlock", post(unlock_node))
        .route("/api/locks", get(list_locks))
        .route("/api/runouts", post(runouts))
        .route("/api/range/parse", post(parse_range))
        .route("/api/save", post(save_solve))
        .route("/api/load", post(load_solve))
        .route("/api/saves", get(list_saves))
        .route("/api/presets", get(get_presets))
        .route("/api/preflop/spot", post(pf_build))
        .route("/api/preflop/estimate", post(pf_estimate))
        .route("/api/preflop/solve", post(pf_solve))
        .route("/api/preflop/stop", post(pf_stop))
        .route("/api/preflop/status", get(pf_status))
        .route("/api/preflop/node", post(pf_node))
        .route("/api/preflop/export", post(pf_export))
        .route("/api/preflop/table", post(pf_table))
        .route("/api/preflop/generate", post(pf_generate))
        .route("/api/preflop/archetypes", get(pf_archetypes))
        .route("/api/preflop/save", post(pf_save_game))
        .route("/api/preflop/load", post(pf_load_game))
        .route("/api/preflop/saves", get(pf_list_games))
        .route("/api/preflop/hero", post(pf_hero))
        .route("/api/preflop/profiles", get(pf_profiles_list))
        .route("/api/preflop/profiles/save", post(pf_profile_save))
        .route("/api/preflop/profiles/get", post(pf_profile_get))
        .route("/api/preflop/lock", post(pf_lock))
        .route("/api/preflop/unlock", post(pf_unlock))
        .fallback_service(serve_dir)
        .with_state(state);

    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3737);
    let addr = format!("127.0.0.1:{port}");
    println!("GTO solver running at http://{addr}");
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
