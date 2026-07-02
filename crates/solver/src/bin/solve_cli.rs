//! Minimal CLI: solve a spot from a JSON config file and print progress.
//!
//! Usage: solve-cli <config.json> [max_iterations] [target_exploit_pct]
//!
//! Env vars:
//!   SOLVER_STORAGE=f32|i16   arena storage (default i16 compressed)
//!   SOLVER_ALGO=dcfr|cfr+|pcfr+   algorithm (default dcfr)
//!   SOLVER_ISO=0             disable suit isomorphism
//!   SOLVER_THREADS=N         rayon thread count (default: rayon's own)
//!   SOLVER_GPU=1             solve on the GPU (requires `--features gpu`
//!                            build; forces f32 storage)

use solver::{Algorithm, RunOptions, Solver, Spot, SpotConfig, Storage};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

fn peak_rss_mb() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|kb| kb.parse::<f64>().ok())
        })
        .map(|kb| kb / 1024.0)
        .unwrap_or(f64::NAN)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: solve-cli <config.json> [max_iterations] [target_exploit_pct]");
        eprintln!("       solve-cli batch <config.json> <boards-file|b1,b2,..> [max_iterations] [target]");
        std::process::exit(1);
    }
    if args[1] == "batch" {
        run_batch(&args[2..]);
        return;
    }
    if let Ok(threads) = std::env::var("SOLVER_THREADS") {
        if let Ok(n) = threads.parse::<usize>() {
            rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build_global()
                .ok();
        }
    }
    let use_gpu = std::env::var("SOLVER_GPU").map(|v| v == "1").unwrap_or(false);
    let storage = if use_gpu {
        Storage::F32
    } else {
        match std::env::var("SOLVER_STORAGE").as_deref() {
            Ok("f32") => Storage::F32,
            _ => Storage::Compressed,
        }
    };
    let algo = std::env::var("SOLVER_ALGO")
        .map(|s| Algorithm::parse(&s).expect("bad SOLVER_ALGO"))
        .unwrap_or(Algorithm::Dcfr);

    let config_text = std::fs::read_to_string(&args[1]).expect("cannot read config");
    let config: SpotConfig = serde_json::from_str(&config_text).expect("invalid config JSON");

    let t0 = std::time::Instant::now();
    let spot = Spot::new(config).expect("failed to build spot");
    let mut solver = Solver::with_storage(Arc::new(spot), storage);
    solver.algo = algo;
    solver.use_isomorphism = std::env::var("SOLVER_ISO").map(|v| v != "0").unwrap_or(true);
    println!(
        "tree built in {:.2}s: {} nodes ({} action nodes), hands {}/{}, arenas {:.1} MB, storage {:?}, algo {:?}",
        t0.elapsed().as_secs_f64(),
        solver.spot.tree.nodes.len(),
        solver.spot.num_action_nodes(),
        solver.spot.hands[0].len(),
        solver.spot.hands[1].len(),
        solver.arena_bytes() as f64 / 1e6,
        storage,
        algo,
    );

    let opts = RunOptions {
        max_iterations: args
            .get(2)
            .and_then(|s| s.parse().ok())
            .unwrap_or(300),
        target_exploit_pct: args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.3),
        check_every: 10,
    };

    if use_gpu {
        run_gpu(&mut solver, &opts);
        return;
    }

    let stop = AtomicBool::new(false);
    let final_progress = solver.run(&opts, &stop, |p| {
        println!(
            "iter {:5}  exploitability {:8.4} chips ({:.3}% pot)  [{:.1}s]",
            p.iteration, p.exploit_chips, p.exploit_pct_pot, p.elapsed_secs
        );
    });
    println!(
        "done: iter {} exploitability {:.3}% pot in {:.1}s  peak_rss {:.0} MB",
        final_progress.iteration,
        final_progress.exploit_pct_pot,
        final_progress.elapsed_secs,
        peak_rss_mb(),
    );
}

/// Batch mode: solve the same tree/range config across many boards, printing
/// one row per board and writing batch_results.json. The basis for
/// multi-flop aggregate analysis.
fn run_batch(rest: &[String]) {
    if rest.len() < 2 {
        eprintln!("usage: solve-cli batch <config.json> <boards-file|b1,b2,..> [max_iterations] [target]");
        std::process::exit(1);
    }
    let use_gpu = cfg!(feature = "gpu")
        && std::env::var("SOLVER_GPU").map(|v| v == "1").unwrap_or(true);
    let save_each = std::env::var("SOLVER_BATCH_SAVE").map(|v| v == "1").unwrap_or(false);
    let config_text = std::fs::read_to_string(&rest[0]).expect("cannot read config");
    let base: SpotConfig = serde_json::from_str(&config_text).expect("invalid config JSON");
    let boards: Vec<String> = match std::fs::read_to_string(&rest[1]) {
        Ok(text) => text
            .split(|c: char| c.is_whitespace() || c == ',')
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect(),
        Err(_) => rest[1].split(',').map(str::to_string).collect(),
    };
    let max_iterations: u32 = rest.get(2).and_then(|s| s.parse().ok()).unwrap_or(300);
    let target: f64 = rest.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.3);
    println!(
        "batch: {} boards, target {target}% pot, max {max_iterations} iters, {}",
        boards.len(),
        if use_gpu { "GPU" } else { "CPU" },
    );

    let t_all = std::time::Instant::now();
    let mut rows = Vec::new();
    for board in &boards {
        let t0 = std::time::Instant::now();
        let mut cfg = base.clone();
        cfg.board = board.clone();
        let pot = cfg.tree.starting_pot;
        let spot = match Spot::new(cfg) {
            Ok(s) => s,
            Err(err) => {
                println!("{board:>8}  ERROR: {err}");
                rows.push(serde_json::json!({"board": board, "error": err}));
                continue;
            }
        };
        let mut solver = Solver::with_storage(
            Arc::new(spot),
            if use_gpu { Storage::F32 } else { Storage::Compressed },
        );
        solver.use_isomorphism =
            std::env::var("SOLVER_ISO").map(|v| v != "0").unwrap_or(true);
        let (iters, pct) = solve_quiet(&mut solver, max_iterations, target, use_gpu);
        let secs = t0.elapsed().as_secs_f64();

        // reach-weighted root EVs per player (pot-share convention)
        solver.ensure_symmetric();
        let view = solver.node_view(&[]).unwrap();
        let avg_ev = |p: usize| -> f64 {
            let (mut n, mut d) = (0f64, 0f64);
            for h in &view.players[p].hands {
                if let Some(ev) = h.ev {
                    n += ev as f64 * h.reach as f64;
                    d += h.reach as f64;
                }
            }
            n / d
        };
        let (ev_oop, ev_ip) = (avg_ev(0), avg_ev(1));
        println!(
            "{board:>8}  iter {iters:4}  exploit {pct:6.3}% pot  EV {ev_oop:6.2}/{ev_ip:6.2}  [{secs:.1}s]"
        );
        rows.push(serde_json::json!({
            "board": board, "iterations": iters, "exploit_pct_pot": pct,
            "ev_oop": ev_oop, "ev_ip": ev_ip, "seconds": secs,
            "pot": pot,
        }));
        if save_each {
            std::fs::create_dir_all("saves").ok();
            let name = format!("saves/batch_{board}.gto");
            if let Err(err) = solver.save(&name) {
                println!("          save failed: {err}");
            }
        }
    }
    std::fs::write(
        "batch_results.json",
        serde_json::to_string_pretty(&rows).unwrap(),
    )
    .expect("cannot write batch_results.json");
    println!(
        "batch done: {} boards in {:.1}s -> batch_results.json",
        boards.len(),
        t_all.elapsed().as_secs_f64()
    );
}

/// Solve to target/max without per-check prints; returns (iterations, exploit%).
fn solve_quiet(solver: &mut Solver, max_iterations: u32, target: f64, use_gpu: bool) -> (u32, f64) {
    #[cfg(feature = "gpu")]
    let pot = solver.spot.tree.config.starting_pot;
    #[cfg(feature = "gpu")]
    if use_gpu {
        let mut gpu = solver::gpu::GpuSolver::new(solver).expect("gpu init failed");
        loop {
            gpu.iterate().expect("gpu iterate failed");
            let check = gpu.iteration % 20 == 0 || gpu.iteration >= max_iterations;
            if check {
                let e = gpu.exploitability(solver).expect("gpu exploitability failed");
                let pct = e / pot * 100.0;
                if pct <= target || gpu.iteration >= max_iterations {
                    gpu.sync_to_cpu(solver).expect("gpu sync failed");
                    return (gpu.iteration, pct);
                }
            }
        }
    }
    let _ = use_gpu;
    let opts = RunOptions {
        max_iterations,
        target_exploit_pct: target,
        check_every: 20,
    };
    let stop = AtomicBool::new(false);
    let p = solver.run(&opts, &stop, |_| {});
    (p.iteration, p.exploit_pct_pot)
}

#[cfg(not(feature = "gpu"))]
fn run_gpu(_solver: &mut Solver, _opts: &RunOptions) {
    eprintln!("SOLVER_GPU=1 requires a build with `--features gpu`");
    std::process::exit(1);
}

#[cfg(feature = "gpu")]
fn run_gpu(solver: &mut Solver, opts: &RunOptions) {
    let pot = solver.spot.tree.config.starting_pot;
    let mut gpu = solver::gpu::GpuSolver::new(solver).expect("gpu init failed");

    if std::env::var("SOLVER_GPU_PROFILE").map(|v| v == "1").unwrap_or(false) {
        // warm up (JIT + caches), then profile a few iterations
        for _ in 0..3 {
            gpu.iterate().expect("gpu iterate failed");
        }
        gpu.synchronize().unwrap();
        let mut total = solver::gpu::Profile::default();
        for _ in 0..5 {
            let p = gpu.iterate_profiled().expect("profile failed");
            for k in 0..7 {
                total.ms[k] += p.ms[k];
                total.launches[k] += p.launches[k];
            }
        }
        let sum: f64 = total.ms.iter().sum();
        println!("--- kernel profile (5 iterations, sync-per-launch) ---");
        for (k, name) in solver::gpu::KERNEL_NAMES.iter().enumerate() {
            println!(
                "{:>12}: {:8.1} ms ({:4.1}%)  {:5} launches",
                name,
                total.ms[k],
                total.ms[k] / sum * 100.0,
                total.launches[k]
            );
        }
        println!("{:>12}: {:8.1} ms", "TOTAL", sum);
        return;
    }
    let start = std::time::Instant::now();
    let mut gpu_secs = 0f64;
    loop {
        let t = std::time::Instant::now();
        gpu.iterate().expect("gpu iterate failed");
        gpu.synchronize().expect("gpu sync failed");
        gpu_secs += t.elapsed().as_secs_f64();
        let check = gpu.iteration % opts.check_every.max(1) == 0
            || gpu.iteration >= opts.max_iterations;
        if check {
            // best response runs on the GPU too — no arena download needed
            let tchk = std::time::Instant::now();
            let e = gpu.exploitability(solver).expect("gpu exploitability failed");
            let chk_ms = tchk.elapsed().as_secs_f64() * 1e3;
            if std::env::var("SOLVER_GPU_TIMECHK").is_ok() {
                println!("   check took {chk_ms:.0} ms");
            }
            let pct = e / pot * 100.0;
            println!(
                "iter {:5}  exploitability {:8.4} chips ({:.3}% pot)  [{:.1}s, gpu {:.1}s]",
                gpu.iteration,
                e,
                pct,
                start.elapsed().as_secs_f64(),
                gpu_secs,
            );
            if pct <= opts.target_exploit_pct || gpu.iteration >= opts.max_iterations {
                gpu.sync_to_cpu(solver).expect("gpu sync failed");
                println!(
                    "done: iter {} exploitability {:.3}% pot in {:.1}s (gpu kernels {:.2}s = {:.0} ms/iter)  peak_rss {:.0} MB",
                    gpu.iteration,
                    pct,
                    start.elapsed().as_secs_f64(),
                    gpu_secs,
                    gpu_secs * 1000.0 / gpu.iteration as f64,
                    peak_rss_mb(),
                );
                break;
            }
        }
    }
}
