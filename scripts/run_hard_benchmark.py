"""
VQE Benchmark on the Genuinely Hard BN Quantum Dot Problem (8 qubits, 4e4o).
Round 3 (Phase G): Statistically honest benchmark execution.

Features:
- Separate evaluations for Initializations:
    (a) 'zero' (theta = 0, deterministic, 1 run)
    (b) 'perturbed' with theta ~ N(0, sigma^2) for sigma in {0.01, 0.05, 0.1, 0.3}, seeds 0-9
    (c) 'random' U(0, 2pi), seeds 0-9
- Iteration budgets:
    L-BFGS-B: maxiter = 300 (ftol 1e-14, gtol 1e-8)
    COBYLA: maxiter = 1000
    GD, ADAM, SPSA, QNSPSA: maxiter = 300
- Learning rate sweep on separate seed set (seeds 100-104), saved to data/lr_sweep_results_hard.json
- Cache Schema v3.0: stores final_params, best_params, iterations_used, hit_maxiter, converged
- Atomic cache writes via temp file and rename
- Parallel execution across physical CPU cores (OMP_NUM_THREADS=1 inside workers)
- Timing probe on 3 configs with projected total time calculation
- Per-run 10-minute timeout guard
- Statistical metrics with bootstrap 95% CI (1000 resamples) for median error
- Plots generated per init-type with log scale and HF gap (8.139 mHa) + 1.6 mHa references
"""
import os
import sys
import json
import time
import shutil
from pathlib import Path

# Enforce single-thread linear algebra inside parallel workers
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector

from src.molecule import load_hard_hamiltonian, DATA_DIR
from src.ansatze import get_ansatz_dict
from src.optimizers import run_vqe_single

CACHE_SCHEMA_VERSION = "3.0"
CACHE_FILE = DATA_DIR / "benchmark_results_hard.json"
LR_SWEEP_FILE = DATA_DIR / "lr_sweep_results_hard.json"
SUMMARY_CSV = DATA_DIR / "hard_benchmark_summary.csv"
FIGURES_DIR = REPO_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Iteration budgets (Phase G.3)
BUDGETS = {
    "L-BFGS-B": 300,
    "COBYLA": 1000,
    "GD": 300,
    "ADAM": 300,
    "SPSA": 300,
    "QNSPSA": 300,
    "ROTOSOLVE": 4
}


from src.optimizers import run_vqe_single, FastStatevectorEstimator


def is_valid_v3_record(r: dict) -> bool:
    """Checks if a cached record adheres to Schema v3.0."""
    required = [
        "ansatz", "init", "optimizer", "final_energy", "final_params",
        "best_params", "total_evaluations", "converged", "hit_maxiter"
    ]
    if not all(k in r for k in required):
        return False
    if r["final_params"] is None or not isinstance(r["final_params"], (list, tuple)):
        return False
    return True


def load_or_init_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cdata = json.load(f)
            if cdata.get("schema_version") == CACHE_SCHEMA_VERSION:
                valid_recs = [r for r in cdata.get("records", []) if is_valid_v3_record(r)]
                return {"schema_version": CACHE_SCHEMA_VERSION, "records": valid_recs}
        except Exception:
            pass
    return {"schema_version": CACHE_SCHEMA_VERSION, "records": []}


def atomic_save_cache(cache_data: dict):
    """Writes cache atomically using temporary file and atomic rename."""
    temp_file = CACHE_FILE.with_suffix(".tmp")
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, indent=2)
    shutil.move(str(temp_file), str(CACHE_FILE))


def run_single_benchmark_worker(
    ansatz_name: str,
    init_name: str,
    optimizer_name: str,
    lr: float | None,
    seed: int,
    sigma: float,
    maxiter: int,
    hamiltonian,
    circuit_decomposed,
    exact_energy: float
) -> dict:
    """Executes a single benchmark run with 10-minute timeout guard."""
    t0 = time.perf_counter()
    estimator = FastStatevectorEstimator(hamiltonian)
    sampler = StatevectorSampler()

    if ansatz_name == "HF_baseline" or circuit_decomposed.num_parameters == 0:
        sv = Statevector(circuit_decomposed)
        H_mat = hamiltonian.to_matrix()
        e_hf = float(np.vdot(sv.data, H_mat.dot(sv.data)).real)
        return {
            "ansatz": ansatz_name,
            "init": init_name,
            "optimizer": optimizer_name,
            "lr": lr,
            "seed": seed,
            "sigma": sigma,
            "final_energy": e_hf,
            "best_energy": e_hf,
            "final_params": [],
            "best_params": [],
            "history": [e_hf],
            "total_evaluations": 1,
            "iterations_used": 1,
            "hit_maxiter": False,
            "converged": True,
            "success": True,
            "status": 0,
            "message": "Hartree-Fock baseline determinant",
            "wall_time": time.perf_counter() - t0,
        }

    try:
        res = run_vqe_single(
            circuit=circuit_decomposed,
            hamiltonian=hamiltonian,
            ansatz_name=ansatz_name,
            init_name=init_name,
            optimizer_name=optimizer_name,
            maxiter=maxiter,
            lr=lr,
            estimator=estimator,
            sampler=sampler,
            seed=seed,
            sigma=sigma
        )
        f_e = float(res["final_energy"])

        # Variational bound check assertion
        if f_e < exact_energy - 1e-9:
            raise RuntimeError(f"Variational bound violated! E={f_e:.10f} < CASCI={exact_energy:.10f}")

        wtime = time.perf_counter() - t0
        stat_val = str(res["status"])
        msg_val = str(res["message"])
        if stat_val == "2" or wtime > 600.0 or "TIMEOUT" in msg_val:
            stat_val = "TIMEOUT"
            hit_maxiter_val = True
            converged_val = False
        else:
            hit_maxiter_val = bool(res["hit_maxiter"])
            converged_val = bool(res["converged"])

        return {
            "ansatz": ansatz_name,
            "init": init_name,
            "optimizer": optimizer_name,
            "lr": lr,
            "seed": seed,
            "sigma": sigma,
            "final_energy": f_e,
            "best_energy": float(res["best_energy"]),
            "final_params": [float(x) for x in res["final_params"]],
            "best_params": [float(x) for x in res["best_params"]],
            "history": [float(h) for h in res.get("energy_history", [])],
            "total_evaluations": int(res["total_evaluations"]),
            "iterations_used": int(res["iterations_used"]),
            "hit_maxiter": hit_maxiter_val,
            "converged": converged_val,
            "success": bool(res["success"]) if stat_val != "TIMEOUT" else False,
            "status": stat_val,
            "message": msg_val,
            "wall_time": wtime,
        }
    except Exception as exc:
        wtime = time.perf_counter() - t0
        return {
            "ansatz": ansatz_name,
            "init": init_name,
            "optimizer": optimizer_name,
            "lr": lr,
            "seed": seed,
            "sigma": sigma,
            "final_energy": 0.0,
            "best_energy": 0.0,
            "final_params": [],
            "best_params": [],
            "history": [],
            "total_evaluations": 0,
            "iterations_used": 0,
            "hit_maxiter": True,
            "converged": False,
            "success": False,
            "status": "ERROR_OR_TIMEOUT",
            "message": str(exc),
            "wall_time": wtime,
        }


def run_timing_probe(ansatze, hamiltonian, exact_energy) -> tuple[float, bool]:
    """Runs a 2-minute timing probe on 3 configurations and estimates total wall time."""
    print("--- Running 2-Minute Timing Probe on 3 Representative Configurations ---")
    probe_configs = [
        ("1-UpCCGSD", "zero", "L-BFGS-B", None, 0, 0.0, 50, BUDGETS["L-BFGS-B"]),
        ("UCCSD", "perturbed_0.05", "COBYLA", None, 0, 0.05, 100, BUDGETS["COBYLA"]),
        ("2-UpCCGSD", "random", "ADAM", 0.05, 0, 0.0, 30, BUDGETS["ADAM"])
    ]
    probe_times = []
    for a_name, i_name, opt_name, lr, s, sig, probe_iter, full_budget in probe_configs:
        t0 = time.time()
        res = run_single_benchmark_worker(
            a_name, i_name, opt_name, lr, s, sig, probe_iter,
            hamiltonian, ansatze[a_name], exact_energy
        )
        dt = time.time() - t0
        scaled_dt = dt * (full_budget / probe_iter)
        probe_times.append(scaled_dt)
        print(f"  Probe config ({a_name}, {opt_name}, {i_name}): measured {dt:.2f}s (scaled full-budget: {scaled_dt:.2f}s), evals = {res['total_evaluations']}")

    mean_t = float(np.mean(probe_times))
    # Approximate number of remaining runs ~ 1500; assume 12 worker processes
    n_cpus = min(os.cpu_count() or 1, 14)
    total_runs_est = 2143
    projected_seconds = (total_runs_est * mean_t) / n_cpus
    projected_hours = projected_seconds / 3600.0

    print(f"\nTiming Probe Mean Time: {mean_t:.2f}s per run across {n_cpus} physical cores.")
    print(f"Projected Total Time for main grid: {projected_hours:.2f} hours ({projected_seconds:.0f} seconds).")

    drop_large = projected_hours > 6.0
    if drop_large:
        print("[NOTICE] Projected total time > 6 hours. Automatically dropping QNSPSA and GD-lr-sweep for 3-UpCCGSD.")
    else:
        print("[NOTICE] Projected total time <= 6 hours. Proceeding with complete benchmark grid.")

    return projected_hours, drop_large


def run_learning_rate_sweep(ansatze, hamiltonian, exact_energy, drop_3up_gd: bool = False) -> dict:
    """Executes learning rate sweep on independent seed set (seeds 100-104)."""
    print("\n--- Phase G.4: Running Learning Rate Sweep (Seeds 100-104) ---")
    sweep_seeds = [100, 101, 102, 103, 104]
    lr_candidates = [0.01, 0.05, 0.1, 0.3]
    ansatz_list = ["DexcG", "PCU2", "UCCSD", "UCCSD-SA", "1-UpCCGSD", "2-UpCCGSD"]
    if not drop_3up_gd:
        ansatz_list.append("3-UpCCGSD")

    sweep_records = []
    if LR_SWEEP_FILE.exists():
        try:
            with open(LR_SWEEP_FILE, "r", encoding="utf-8") as f:
                sweep_records = json.load(f).get("sweep_records", [])
        except Exception:
            pass

    existing_keys = set((r["ansatz"], r["optimizer"], r["lr"], r["seed"]) for r in sweep_records)

    tasks = []
    for a_name in ansatz_list:
        for opt in ["GD", "ADAM"]:
            if a_name == "3-UpCCGSD" and opt == "GD" and drop_3up_gd:
                continue
            for lr in lr_candidates:
                for s in sweep_seeds:
                    if (a_name, opt, lr, s) not in existing_keys:
                        tasks.append((a_name, opt, lr, s))

    if tasks:
        print(f"Running {len(tasks)} learning rate sweep evaluations in parallel...")
        n_cpus = min(os.cpu_count() or 1, 14)
        batch_size = 28
        t_sw0 = time.time()
        for i in range(0, len(tasks), batch_size):
            chunk = tasks[i:i + batch_size]
            results = Parallel(n_jobs=n_cpus)(
                delayed(run_single_benchmark_worker)(
                    a_name, "random", opt, lr, s, 0.0, 25,
                    hamiltonian, ansatze[a_name], exact_energy
                )
                for a_name, opt, lr, s in chunk
            )
            for res in results:
                sweep_records.append({
                    "ansatz": res["ansatz"],
                    "optimizer": res["optimizer"],
                    "lr": res["lr"],
                    "seed": res["seed"],
                    "final_energy": res["final_energy"],
                    "error_mHa": abs(res["final_energy"] - exact_energy) * 1000.0,
                    "total_evaluations": res["total_evaluations"],
                    "converged": res["converged"]
                })

            with open(LR_SWEEP_FILE, "w", encoding="utf-8") as f:
                json.dump({"sweep_records": sweep_records}, f, indent=2)

            done = min(i + batch_size, len(tasks))
            el = time.time() - t_sw0
            rem = (len(tasks) - done) / (done / el) if done > 0 and el > 0 else 0.0
            print(f"  [LR Sweep {done}/{len(tasks)}] completed | Elapsed: {el:.1f}s | ETA: {rem:.1f}s")

    # Select best lr by median error on seeds 100-104
    df_sw = pd.DataFrame(sweep_records)
    best_lr_map = {}
    for a_name in ["DexcG", "PCU2", "UCCSD", "UCCSD-SA", "1-UpCCGSD", "2-UpCCGSD", "3-UpCCGSD"]:
        for opt in ["GD", "ADAM"]:
            sub = df_sw[(df_sw["ansatz"] == a_name) & (df_sw["optimizer"] == opt)]
            if not sub.empty:
                meds = sub.groupby("lr")["error_mHa"].median()
                best_lr = float(meds.idxmin())
                best_lr_map[(a_name, opt)] = best_lr
            else:
                best_lr_map[(a_name, opt)] = 0.05

    print("Optimal learning rates selected from independent seeds (100-104):")
    for (a, o), val in best_lr_map.items():
        print(f"  {a:12s} {o:4s}: lr = {val}")

    return best_lr_map


def execute_full_statistically_honest_benchmark():
    print("=" * 80)
    print("PHASE G: EXECUTING FULL STATISTICALLY HONEST BENCHMARK GRID")
    print("=" * 80)

    hamiltonian, exact_energy, meta = load_hard_hamiltonian()
    ref_det_energy = float(meta["reference_determinant_energy"])
    delta_corr = abs(ref_det_energy - exact_energy)

    # Pre-decompose all ansatze
    ansatze_raw = get_ansatz_dict(
        num_spatial_orbitals=4,
        num_particles=(2, 2),
        pcu2_reps=1,
        include_sa=True,
        include_k_variants=True,
        include_hf=True
    )
    ansatze = {}
    for name, qc in ansatze_raw.items():
        if qc.num_parameters == 0:
            ansatze[name] = qc
        else:
            ansatze[name] = qc.decompose().decompose()

    # Timing probe
    _, drop_large = run_timing_probe(ansatze, hamiltonian, exact_energy)

    # Learning rate sweep
    best_lr_map = run_learning_rate_sweep(ansatze, hamiltonian, exact_energy, drop_3up_gd=drop_large)

    # Main Grid Specification
    ansatz_order = [
        "HF_baseline", "DexcG", "PCU2", "UCCSD", "UCCSD-SA",
        "1-UpCCGSD", "2-UpCCGSD", "3-UpCCGSD"
    ]
    optimizers = ["COBYLA", "L-BFGS-B", "ADAM", "GD", "SPSA"]
    if not drop_large:
        optimizers.append("QNSPSA")

    seeds = list(range(10))
    perturbed_sigmas = [0.01, 0.05, 0.1, 0.3]

    cache = load_or_init_cache()
    records = cache.get("records", [])
    cache_keys = set(
        (r["ansatz"], r["init"], r["optimizer"], str(r.get("lr")), r["seed"])
        for r in records
    )
    print(f"\nExisting valid Schema v3.0 records in cache: {len(records)}")

    # Build task list
    tasks = []
    # 1. HF Baseline (1 run)
    if ("HF_baseline", "zero", "none", "None", 0) not in cache_keys:
        tasks.append(("HF_baseline", "zero", "none", None, 0, 0.0, 1))

    # 2. Parameterized Ansatze ordered by optimizer priority for maximum throughput
    opt_priority = ["L-BFGS-B", "COBYLA", "ROTOSOLVE", "SPSA", "QNSPSA", "ADAM", "GD"]
    for opt in opt_priority:
        if opt not in optimizers and opt != "ROTOSOLVE":
            continue
        for a_name in ansatz_order:
            if a_name == "HF_baseline":
                continue
            if opt == "ROTOSOLVE" and a_name != "DexcG":
                continue
            if a_name == "3-UpCCGSD" and opt == "QNSPSA" and drop_large:
                continue

            lr_val = best_lr_map.get((a_name, opt), None)
            budget = BUDGETS.get(opt, 300)

            # (a) Zero init (1 deterministic run, seed 0)
            k_zero = (a_name, "zero", opt, str(lr_val), 0)
            if k_zero not in cache_keys:
                tasks.append((a_name, "zero", opt, lr_val, 0, 0.0, budget))

            # (b) Perturbed inits across sigmas and 10 seeds
            for sig in perturbed_sigmas:
                i_name = f"perturbed_{sig}"
                for s in seeds:
                    k_pert = (a_name, i_name, opt, str(lr_val), s)
                    if k_pert not in cache_keys:
                        tasks.append((a_name, i_name, opt, lr_val, s, sig, budget))

            # (c) Random init across 10 seeds
            for s in seeds:
                k_rand = (a_name, "random", opt, str(lr_val), s)
                if k_rand not in cache_keys:
                    tasks.append((a_name, "random", opt, lr_val, s, 0.0, budget))

    print(f"Total new runs to execute: {len(tasks)}")

    if tasks:
        n_cpus = min(os.cpu_count() or 1, 14)
        print(f"Executing tasks in parallel across {n_cpus} physical cores...")
        t_start = time.time()

        batch_size = n_cpus
        for i in range(0, len(tasks), batch_size):
            chunk = tasks[i:i + batch_size]
            results = Parallel(n_jobs=n_cpus)(
                delayed(run_single_benchmark_worker)(
                    a_name, i_name, opt, lr, s, sig, miter,
                    hamiltonian, ansatze[a_name], exact_energy
                )
                for a_name, i_name, opt, lr, s, sig, miter in chunk
            )
            for res in results:
                records.append(res)
                k = (res["ansatz"], res["init"], res["optimizer"], str(res.get("lr")), res["seed"])
                cache_keys.add(k)

            atomic_save_cache({"schema_version": CACHE_SCHEMA_VERSION, "records": records})
            completed = min(i + batch_size, len(tasks))
            elapsed = time.time() - t_start
            rate = completed / elapsed if elapsed > 0 else 1.0
            remaining_time = (len(tasks) - completed) / rate if rate > 0 else 0.0
            print(f"  [{completed}/{len(tasks)}] runs completed | Elapsed: {elapsed/60:.1f}m | ETA: {remaining_time/60:.1f}m")

    print("\nBenchmark grid execution finished successfully!")

    # Process metrics and generate summary CSV
    process_benchmark_metrics_and_figures(records, exact_energy, ref_det_energy, delta_corr)


def process_benchmark_metrics_and_figures(records: list[dict], exact_energy: float, ref_energy: float, delta_corr: float):
    print("\n--- Processing Metrics and Generating Summary Outputs ---")
    processed = []
    for r in records:
        f_e = r["final_energy"]
        err_mha = abs(f_e - exact_energy) * 1000.0
        pct_corr = ((ref_energy - f_e) / delta_corr) * 100.0 if delta_corr > 1e-12 else 0.0
        better_than_hf = (f_e < ref_energy - 1e-9)
        success = (err_mha < 1.6 and better_than_hf)

        # Classify init type group
        init_raw = r["init"]
        if init_raw.startswith("perturbed"):
            init_group = "perturbed"
        else:
            init_group = init_raw

        processed.append({
            "ansatz": r["ansatz"],
            "init": init_raw,
            "init_group": init_group,
            "optimizer": r["optimizer"],
            "lr": r.get("lr"),
            "seed": r["seed"],
            "final_energy": f_e,
            "error_mHa": err_mha,
            "pct_corr_recovered": pct_corr,
            "better_than_HF": better_than_hf,
            "success": success,
            "converged": r["converged"],
            "hit_maxiter": r["hit_maxiter"],
            "total_evaluations": r["total_evaluations"],
            "wall_time": r["wall_time"],
        })

    df = pd.DataFrame(processed)

    # Group by (init, ansatz, optimizer)
    summary_rows = []
    groups = df.groupby(["init", "ansatz", "optimizer"])
    for (init_name, a_name, opt_name), group in groups:
        errors = group["error_mHa"].values
        pcts = group["pct_corr_recovered"].values
        evals = group["total_evaluations"].values
        convs = group["converged"].values
        succs = group["success"].values
        betters = group["better_than_HF"].values
        times = group["wall_time"].values

        n_runs = len(errors)
        q25, med, q75 = np.percentile(errors, [25, 50, 75])
        iqr = q75 - q25
        min_err = np.min(errors)
        max_err = np.max(errors)
        med_pct = np.median(pcts)
        med_evals = np.median(evals)
        frac_conv = np.mean(convs) * 100.0
        succ_rate = np.mean(succs) * 100.0
        better_rate = np.mean(betters) * 100.0
        mean_time = np.mean(times)

        # Bootstrap 95% CI for median
        if n_runs > 1:
            rng = np.random.RandomState(42)
            boot_medians = [
                np.median(rng.choice(errors, size=n_runs, replace=True))
                for _ in range(1000)
            ]
            ci_lower = float(np.percentile(boot_medians, 2.5))
            ci_upper = float(np.percentile(boot_medians, 97.5))
        else:
            ci_lower = float(med)
            ci_upper = float(med)

        summary_rows.append({
            "Init_Type": init_name,
            "Ansatz": a_name,
            "Optimizer": opt_name,
            "N_Runs": n_runs,
            "Median_Error_mHa": float(med),
            "IQR_Error_mHa": float(iqr),
            "CI_95_Lower_mHa": ci_lower,
            "CI_95_Upper_mHa": ci_upper,
            "Min_Error_mHa": float(min_err),
            "Max_Error_mHa": float(max_err),
            "Median_Pct_Corr": float(med_pct),
            "Success_Rate_pct": float(succ_rate),
            "Better_than_HF_pct": float(better_rate),
            "Fraction_Converged_pct": float(frac_conv),
            "Median_Evaluations": float(med_evals),
            "Mean_Wall_Time_s": float(mean_time)
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.sort_values(by=["Init_Type", "Median_Error_mHa"], inplace=True)
    df_summary.to_csv(SUMMARY_CSV, index=False)
    print(f"Saved comprehensive summary to {SUMMARY_CSV} ({len(df_summary)} rows).")

    # Generate Figures
    generate_benchmark_figures(df)


def generate_benchmark_figures(df: pd.DataFrame):
    """Generates the three benchmark figures with log-scale and reference lines."""
    hf_gap = 8.1392
    chem_acc = 1.6000

    # 1. Error vs Evaluations per init type
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    init_types = ["zero", "perturbed_0.05", "random"]
    titles = ["Zero Initialization (Deterministic)", "Perturbed (sigma=0.05)", "Random U(0, 2pi)"]

    for ax, itype, title in zip(axes, init_types, titles):
        sub = df[df["init"] == itype]
        for a_name in sub["ansatz"].unique():
            a_sub = sub[sub["ansatz"] == a_name]
            if not a_sub.empty:
                med_eval = a_sub.groupby("optimizer")["total_evaluations"].median()
                med_err = a_sub.groupby("optimizer")["error_mHa"].median()
                ax.scatter(med_eval, med_err, label=a_name, s=60, alpha=0.85)

        ax.axhline(hf_gap, color="red", linestyle="--", linewidth=1.5, label="HF Gap (8.139 mHa)")
        ax.axhline(chem_acc, color="green", linestyle=":", linewidth=1.5, label="Chemical Acc (1.6 mHa)")
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Median Evaluations", fontsize=10)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)

    axes[0].set_ylabel("Error vs CASCI (mHa, log-scale)", fontsize=11)
    axes[2].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    fig_path = FIGURES_DIR / "hard_error_vs_evals.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Generated {fig_path}")

    # 2. Boxplots of Error per Optimizer
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, itype, title in zip(axes, init_types, titles):
        sub = df[df["init"] == itype]
        if not sub.empty:
            sub.boxplot(column="error_mHa", by="optimizer", ax=ax, grid=False)
        ax.axhline(hf_gap, color="red", linestyle="--", linewidth=1.2, label="HF Gap (8.139 mHa)")
        ax.axhline(chem_acc, color="green", linestyle=":", linewidth=1.2, label="Chemical Acc (1.6 mHa)")
        ax.set_yscale("log")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Optimizer", fontsize=10)

    axes[0].set_ylabel("Error vs CASCI (mHa, log-scale)", fontsize=11)
    fig.suptitle("Hard Benchmark VQE Error Distributions per Optimizer", fontsize=13, y=1.02)
    plt.tight_layout()
    fig_path2 = FIGURES_DIR / "hard_error_boxplots.png"
    plt.savefig(fig_path2, dpi=300)
    plt.close()
    print(f"Generated {fig_path2}")

    # 3. Heatmap of Median % Correlation Recovered
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for ax, itype, title in zip(axes, init_types, titles):
        sub = df[df["init"] == itype]
        if not sub.empty:
            pivot = sub.pivot_table(index="ansatz", columns="optimizer", values="pct_corr_recovered", aggfunc="median")
            im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            plt.colorbar(im, ax=ax, label="Median % Corr Recovered")
        ax.set_title(f"Median % Corr ({title})", fontsize=11, fontweight="bold")

    plt.tight_layout()
    fig_path3 = FIGURES_DIR / "hard_pct_corr_heatmap.png"
    plt.savefig(fig_path3, dpi=300)
    plt.close()
    print(f"Generated {fig_path3}")


if __name__ == "__main__":
    execute_full_statistically_honest_benchmark()
