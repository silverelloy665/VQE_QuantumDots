"""
VQE Benchmark on the Genuinely Hard BN Quantum Dot Problem (8 qubits, 4e4o).
Executes Phase C benchmark grid across:
- Ansatze: HF_baseline, DexcG, PCU2, UCCSD, UCCSD-SA, 1-UpCCGSD, 2-UpCCGSD, 3-UpCCGSD
- Optimizers: ADAM, GD (lr sweep {0.01, 0.05, 0.1, 0.3}), SPSA, QNSPSA, L-BFGS-B, COBYLA
- 10 seeds (0..9) for random init and stochastic optimizers
- Incremental JSON cache in data/benchmark_results_hard.json
- Figures in figures/ (error vs evals, boxplots, heatmap)
- Summary table in data/hard_benchmark_summary.csv
"""
import os
import sys
import json
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.quantum_info import Statevector

from src.molecule import load_hard_hamiltonian, DATA_DIR
from src.ansatze import get_ansatz_dict
from src.optimizers import (
    get_initial_point,
    compute_gradient_batched,
    run_vqe_single
)

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = DATA_DIR / "benchmark_results_hard.json"
SUMMARY_CSV = DATA_DIR / "hard_benchmark_summary.csv"


class FastStatevectorEstimator:
    """
    Accelerated exact Statevector estimator for 8-qubit circuits.
    Evaluates expectations using matrix-vector multiplication in < 1 ms.
    Maintains exact Qiskit Primitive interface.
    """
    def __init__(self, hamiltonian):
        self.H_mat = hamiltonian.to_matrix()

    def run(self, pubs):
        results = []
        for qc, _, param_list in pubs:
            evs = []
            for p in param_list:
                bound = qc.assign_parameters(p)
                sv = Statevector(bound)
                v = float(np.vdot(sv.data, self.H_mat.dot(sv.data)).real)
                evs.append(v)
            class _Data:
                def __init__(self, vals):
                    self.evs = np.array(vals, dtype=float)
            class _Res:
                def __init__(self, vals):
                    self.data = _Data(vals)
            results.append(_Res(evs))
        class _Job:
            def __init__(self, res):
                self._res = res
            def result(self):
                return self._res
        return _Job(results)


def load_or_init_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"records": []}


def save_cache(cache_data: dict):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, indent=2)


def run_single_benchmark_run(
    circuit,
    hamiltonian,
    ansatz_name: str,
    init_name: str,
    optimizer_name: str,
    lr: float | None,
    seed: int,
    estimator,
    sampler,
    maxiter: int = 30
) -> dict:
    t0 = time.perf_counter()
    n_params = circuit.num_parameters

    if ansatz_name == "HF_baseline" or n_params == 0:
        # 0 parameters: evaluate Hartree-Fock reference determinant
        bound = circuit
        sv = Statevector(bound)
        H_mat = hamiltonian.to_matrix()
        e_hf = float(np.vdot(sv.data, H_mat.dot(sv.data)).real)
        return {
            "ansatz": ansatz_name,
            "init": init_name,
            "optimizer": optimizer_name,
            "lr": lr,
            "seed": seed,
            "final_energy": e_hf,
            "history": [e_hf],
            "total_evaluations": 1,
            "wall_time": time.perf_counter() - t0,
            "converged": True,
        }

    res = run_vqe_single(
        circuit=circuit,
        hamiltonian=hamiltonian,
        ansatz_name=ansatz_name,
        init_name=init_name,
        optimizer_name=optimizer_name,
        maxiter=maxiter,
        lr=lr,
        estimator=estimator,
        sampler=sampler,
        seed=seed
    )

    return {
        "ansatz": ansatz_name,
        "init": init_name,
        "optimizer": optimizer_name,
        "lr": lr,
        "seed": seed,
        "final_energy": float(res["final_energy"]),
        "history": [float(h) for h in res.get("energy_history", res.get("convergence_history", []))],
        "total_evaluations": int(res["total_evaluations"]),
        "wall_time": float(res["wall_time"]),
        "converged": bool(res.get("converged", True)),
    }


def execute_hard_benchmark(maxiter: int = 30):
    print("=== Loading Hard Problem Benchmark (8 qubits, 4e4o) ===")
    hamiltonian, exact_energy, meta = load_hard_hamiltonian()
    ref_det_energy = float(meta["reference_determinant_energy"])
    delta_corr = abs(ref_det_energy - exact_energy)
    gap_mHa = delta_corr * 1000.0

    print(f"Molecule: {meta['label']} ({meta['molecule']})")
    print(f"Scale: {meta.get('scale', 1.4)}, Basis: {meta['basis']}, Orbital Selection: {meta.get('orbital_selection', 'natural_orbitals')}")
    print(f"Active Space: ({meta['n_active_electrons']}e, {meta['n_active_orbitals']}o) -> {meta['num_qubits']} qubits")
    print(f"Reference Energy: {ref_det_energy:.8f} Ha | Exact CASCI: {exact_energy:.8f} Ha | Gap: {gap_mHa:.4f} mHa")
    print(f"Variational Bound Sanity check: CASCI = {exact_energy:.8f} Ha\n")

    # Fast statevector estimator
    estimator = FastStatevectorEstimator(hamiltonian)
    sampler = StatevectorSampler()

    # Build and pre-decompose ansatze
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

    print("Ansätze parameter counts:")
    for name, qc in ansatze.items():
        print(f"  {name:12s}: {qc.num_parameters:2d} parameters")

    ansatz_order = [
        "HF_baseline",
        "DexcG",
        "PCU2",
        "UCCSD",
        "UCCSD-SA",
        "1-UpCCGSD",
        "2-UpCCGSD",
        "3-UpCCGSD"
    ]

    seeds = list(range(10))
    gd_lr_candidates = [0.01, 0.05, 0.1, 0.3]

    cache = load_or_init_cache()
    records = cache.get("records", [])
    cache_keys = set(
        (r["ansatz"], r["init"], r["optimizer"], str(r.get("lr")), r["seed"])
        for r in records
    )

    print(f"\nExisting cached records: {len(records)}")

    # 1. Sweep GD and ADAM over learning rates on representative seed 0
    print("\n--- Running Learning Rate Sweep on GD and ADAM (lr in {0.01, 0.05, 0.1, 0.3}) ---")
    for lr in gd_lr_candidates:
        for opt in ["GD", "ADAM"]:
            key = ("DexcG", "random", opt, str(lr), 0)
            if key not in cache_keys:
                res = run_single_benchmark_run(
                    ansatze["DexcG"], hamiltonian, "DexcG", "random", opt, lr, 0,
                    estimator, sampler, maxiter=maxiter
                )
                records.append(res)
                cache_keys.add(key)
                save_cache({"records": records})
                err_mha = abs(res["final_energy"] - exact_energy) * 1000.0
                print(f"  [LR Sweep] {opt:4s} lr={lr:<4.2f} seed=0: error={err_mha:7.4f} mHa, evals={res['total_evaluations']}")

    # Pick best learning rate for GD and ADAM from sweep
    lr_sweep_records = [r for r in records if r["ansatz"] == "DexcG" and r["seed"] == 0 and r["lr"] is not None]
    best_lr = {}
    for opt in ["GD", "ADAM"]:
        sub = [r for r in lr_sweep_records if r["optimizer"] == opt]
        if sub:
            best_sub = min(sub, key=lambda x: abs(x["final_energy"] - exact_energy))
            best_lr[opt] = best_sub["lr"]
        else:
            best_lr[opt] = 0.05
    print(f"Selected best learning rates: GD={best_lr['GD']}, ADAM={best_lr['ADAM']}\n")

    # 2. Main Benchmark Grid across all 8 ansatze x 6 optimizers x 10 seeds x inits
    # Inits: "random" across 10 seeds (0..9); also run "zero" init for seed 0
    optimizers = ["COBYLA", "L-BFGS-B", "ADAM", "GD", "SPSA", "QNSPSA"]

    print("--- Running Main Benchmark Grid (10 seeds, statevector exact) ---")
    run_count = 0
    for a_name in ansatz_order:
        qc = ansatze[a_name]

        if a_name == "HF_baseline":
            # Baseline only needs 1 run
            key = (a_name, "none", "none", "None", 0)
            if key not in cache_keys:
                res = run_single_benchmark_run(
                    qc, hamiltonian, a_name, "none", "none", None, 0,
                    estimator, sampler, maxiter=1
                )
                records.append(res)
                cache_keys.add(key)
                save_cache({"records": records})
            continue

        for opt in optimizers:
            lr_val = best_lr.get(opt, None)

            # (a) Zero init (seed 0)
            key_zero = (a_name, "zero", opt, str(lr_val), 0)
            if key_zero not in cache_keys:
                res = run_single_benchmark_run(
                    qc, hamiltonian, a_name, "zero", opt, lr_val, 0,
                    estimator, sampler, maxiter=maxiter
                )
                # Sanity check: variational bound
                if res["final_energy"] < exact_energy - 1e-9:
                    raise RuntimeError(f"Variational bound violated! E={res['final_energy']:.10f} < CASCI={exact_energy:.10f}")
                records.append(res)
                cache_keys.add(key_zero)
                save_cache({"records": records})
                run_count += 1
                err_mha = abs(res["final_energy"] - exact_energy) * 1000.0
                print(f"[{run_count:3d}] {a_name:10s} zero  {opt:8s}: error={err_mha:7.4f} mHa, evals={res['total_evaluations']:4d}, time={res['wall_time']:.2f}s")

            # (b) Random init across 10 seeds
            for s in seeds:
                key_rand = (a_name, "random", opt, str(lr_val), s)
                if key_rand not in cache_keys:
                    res = run_single_benchmark_run(
                        qc, hamiltonian, a_name, "random", opt, lr_val, s,
                        estimator, sampler, maxiter=maxiter
                    )
                    if res["final_energy"] < exact_energy - 1e-9:
                        raise RuntimeError(f"Variational bound violated! E={res['final_energy']:.10f} < CASCI={exact_energy:.10f}")
                    records.append(res)
                    cache_keys.add(key_rand)
                    save_cache({"records": records})
                    run_count += 1
                    err_mha = abs(res["final_energy"] - exact_energy) * 1000.0
                    print(f"[{run_count:3d}] {a_name:10s} rand(s={s}) {opt:8s}: error={err_mha:7.4f} mHa, evals={res['total_evaluations']:4d}, time={res['wall_time']:.2f}s")

    print(f"\n[OK] Benchmark grid complete! Total records in cache: {len(records)}")

    # 3. Process metrics and aggregate results
    processed = []
    for r in records:
        f_e = r["final_energy"]
        err_mha = abs(f_e - exact_energy) * 1000.0
        # Variational check
        assert f_e >= exact_energy - 1e-9, f"Variational bound violated for {r['ansatz']}!"
        pct_corr = ((ref_det_energy - f_e) / delta_corr) * 100.0 if delta_corr > 1e-12 else 0.0
        better_than_hf = (f_e < ref_det_energy - 1e-9)
        success = (err_mha < 1.6 and better_than_hf)

        processed.append({
            "ansatz": r["ansatz"],
            "init": r["init"],
            "optimizer": r["optimizer"],
            "lr": r.get("lr"),
            "seed": r["seed"],
            "final_energy": f_e,
            "error_mHa": err_mha,
            "pct_corr_recovered": pct_corr,
            "better_than_HF": better_than_hf,
            "success": success,
            "total_evaluations": r["total_evaluations"],
            "wall_time": r["wall_time"],
        })

    df = pd.DataFrame(processed)

    # 4. Aggregate across random init seeds (seeds 0..9)
    df_rand = df[df["init"] == "random"].copy()
    grouped = df_rand.groupby(["ansatz", "optimizer"])

    summary_records = []
    for (a_name, opt_name), group in grouped:
        errors = group["error_mHa"].values
        pcts = group["pct_corr_recovered"].values
        evals = group["total_evaluations"].values
        times = group["wall_time"].values
        better = group["better_than_HF"].values
        succ = group["success"].values

        q25, med, q75 = np.percentile(errors, [25, 50, 75])
        iqr = q75 - q25

        summary_records.append({
            "Ansatz": a_name,
            "Optimizer": opt_name,
            "N_Seeds": len(group),
            "Median_Error_mHa": float(med),
            "IQR_Error_mHa": float(iqr),
            "Min_Error_mHa": float(np.min(errors)),
            "Max_Error_mHa": float(np.max(errors)),
            "Mean_Pct_Corr": float(np.mean(pcts)),
            "Median_Evaluations": float(np.median(evals)),
            "Mean_Wall_Time_s": float(np.mean(times)),
            "Success_Rate_pct": float(np.mean(succ) * 100.0),
            "Better_than_HF_pct": float(np.mean(better) * 100.0),
        })

    df_sum = pd.DataFrame(summary_records)
    # Sort ranking: primary by success rate descending, then median error ascending
    df_sum.sort_values(by=["Success_Rate_pct", "Median_Error_mHa"], ascending=[False, True], inplace=True)
    df_sum.to_csv(SUMMARY_CSV, index=False)
    print(f"\n[OK] Saved aggregated benchmark summary to {SUMMARY_CSV}")

    # Print ranking table
    print("\n=== HARD BENCHMARK RANKING (Random Init, 10 Seeds) ===")
    print(df_sum[["Ansatz", "Optimizer", "Median_Error_mHa", "IQR_Error_mHa", "Min_Error_mHa", "Mean_Pct_Corr", "Success_Rate_pct", "Median_Evaluations"]].to_string(index=False))

    # 5. Generate Plots in figures/
    generate_benchmark_plots(records, df, df_sum, exact_energy, ref_det_energy)


def generate_benchmark_plots(records, df, df_sum, exact_energy, ref_det_energy):
    print("\n--- Generating Benchmark Figures in figures/ ---")

    # Plot 1: Error vs evaluations (median + IQR band) per ansatz
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ansatz_list = ["DexcG", "PCU2", "UCCSD", "UCCSD-SA", "1-UpCCGSD", "2-UpCCGSD", "3-UpCCGSD"]
    colors = plt.cm.tab10.colors

    for idx, a_name in enumerate(ansatz_list):
        # Gather all convergence histories for this ansatz across random starts
        sub_records = [r for r in records if r["ansatz"] == a_name and r["init"] == "random"]
        if not sub_records:
            continue
        max_len = max(len(r["history"]) for r in sub_records)
        padded = []
        for r in sub_records:
            h = r["history"]
            if len(h) < max_len:
                h = h + [h[-1]] * (max_len - len(h))
            padded.append(h)
        arr = np.array(padded)
        err_arr = np.abs(arr - exact_energy) * 1000.0  # mHa

        med = np.median(err_arr, axis=0)
        q25 = np.percentile(err_arr, 25, axis=0)
        q75 = np.percentile(err_arr, 75, axis=0)
        iters = np.arange(1, len(med) + 1)

        c = colors[idx % len(colors)]
        ax.plot(iters, med, label=a_name, color=c, lw=2)
        ax.fill_between(iters, q25, q75, color=c, alpha=0.15)

    ax.axhline(1.6, color="black", ls="--", lw=1.2, label="Chemical Accuracy (1.6 mHa)")
    hf_err = (ref_det_energy - exact_energy) * 1000.0
    ax.axhline(hf_err, color="gray", ls=":", lw=1.2, label=f"HF Baseline ({hf_err:.2f} mHa)")

    ax.set_yscale("log")
    ax.set_xlabel("VQE Iteration", fontsize=11)
    ax.set_ylabel("Error vs CASCI (mHa, log scale)", fontsize=11)
    ax.set_title("VQE Optimization Trajectory on Hard Benchmark (Median + IQR)", fontsize=12, pad=10)
    ax.legend(frameon=True, fontsize=9, loc="upper right")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    fig.tight_layout()
    p1 = FIGURES_DIR / "hard_error_vs_evals.png"
    fig.savefig(p1, dpi=300)
    plt.close(fig)
    print(f"  [OK] Saved {p1}")

    # Plot 2: Boxplots of error per optimizer
    fig, ax = plt.subplots(figsize=(8, 5))
    df_rand = df[df["init"] == "random"].copy()
    opts = ["COBYLA", "L-BFGS-B", "ADAM", "GD", "SPSA", "QNSPSA"]
    box_data = [df_rand[df_rand["optimizer"] == opt]["error_mHa"].values for opt in opts]

    bp = ax.boxplot(box_data, labels=opts, patch_artist=True, showmeans=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#a6cee3")
        patch.set_edgecolor("#1f78b4")

    ax.axhline(1.6, color="crimson", ls="--", lw=1.2, label="Chemical Accuracy (1.6 mHa)")
    ax.axhline(hf_err, color="gray", ls=":", lw=1.2, label=f"HF Baseline ({hf_err:.2f} mHa)")
    ax.set_yscale("log")
    ax.set_ylabel("Error vs CASCI (mHa, log scale)", fontsize=11)
    ax.set_xlabel("Optimizer", fontsize=11)
    ax.set_title("Optimizer Performance Distribution Across All Ansätze (10 Seeds)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", ls="--", alpha=0.5)
    fig.tight_layout()
    p2 = FIGURES_DIR / "hard_error_boxplots.png"
    fig.savefig(p2, dpi=300)
    plt.close(fig)
    print(f"  [OK] Saved {p2}")

    # Plot 3: Heatmap of % correlation recovered (Ansatz x Optimizer)
    pivot = df_sum.pivot(index="Ansatz", columns="Optimizer", values="Mean_Pct_Corr")
    # Reindex for clean presentation
    ansatz_rows = [a for a in ["DexcG", "PCU2", "UCCSD", "UCCSD-SA", "1-UpCCGSD", "2-UpCCGSD", "3-UpCCGSD"] if a in pivot.index]
    opt_cols = [o for o in ["COBYLA", "L-BFGS-B", "ADAM", "GD", "SPSA", "QNSPSA"] if o in pivot.columns]
    pivot = pivot.loc[ansatz_rows, opt_cols]

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, cmap="YlGnBu", aspect="auto", vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, label="% Correlation Energy Recovered")

    ax.set_xticks(np.arange(len(opt_cols)))
    ax.set_yticks(np.arange(len(ansatz_rows)))
    ax.set_xticklabels(opt_cols, fontsize=10)
    ax.set_yticklabels(ansatz_rows, fontsize=10)

    for i in range(len(ansatz_rows)):
        for j in range(len(opt_cols)):
            val = pivot.values[i, j]
            text_color = "white" if val > 60 else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", color=text_color, fontweight="bold", fontsize=9)

    ax.set_title("Correlation Energy Recovered (%): Ansatz vs. Optimizer", fontsize=12, pad=12)
    fig.tight_layout()
    p3 = FIGURES_DIR / "hard_pct_corr_heatmap.png"
    fig.savefig(p3, dpi=300)
    plt.close(fig)
    print(f"  [OK] Saved {p3}")


if __name__ == "__main__":
    execute_hard_benchmark(maxiter=25)

