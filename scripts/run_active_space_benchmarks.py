"""
Strictly serial multi-orbital active-space VQE benchmarks for B8N8H10.

Constraints:
  1. Strictly serial execution (synchronous for-loop, no multiprocessing/threads).
  2. Hard-skips k-UpCCGSD at 12 qubits (6e6o) to prevent 855-parameter binding bottleneck.
  3. Forces SPSA-only for 6e6o active space (unless --exhaustive-6e6o is passed).
  4. Real-time tqdm updates with pbar.set_postfix_str() and pbar.write().
  5. CLI flags: --skip-6e6o, --skip-excel, --force-4e4o, --force-6e6o.
"""
import sys
import os
import time
import json
import argparse
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from tqdm import tqdm

from qiskit.circuit import QuantumCircuit
from qiskit.primitives import StatevectorEstimator, StatevectorSampler

from src.molecule import load_or_build_bn_dot_hamiltonian
from src.ansatze import get_ansatz_dict
from src.optimizers import run_vqe_single
from src.benchmark import run_full_vqe_benchmark

DATA_DIR = REPO_ROOT / "data"


def get_active_space_ansatze(n_electrons: int, n_orbitals: int) -> dict[str, QuantumCircuit]:
    """
    Builds decomposed ansatz circuits for the active space.
    If space is (6, 6), skips k-UpCCGSD entirely to avoid the 855-parameter overhead.
    """
    n_alpha = n_electrons // 2
    n_beta = n_electrons - n_alpha
    particles = (n_alpha, n_beta)

    # For 6e6o, explicitly construct only DexcG, PCU2, and UCCSD
    if (n_electrons, n_orbitals) == (6, 6):
        from qiskit_nature.second_q.mappers import JordanWignerMapper
        from qiskit_nature.second_q.circuit.library import HartreeFock, UCC
        from src.ansatze import build_particle_conserving_u2

        mapper = JordanWignerMapper()
        hf_state = HartreeFock(n_orbitals, particles, mapper)

        dexcg = UCC(
            num_spatial_orbitals=n_orbitals,
            num_particles=particles,
            excitations="d",
            qubit_mapper=mapper,
            initial_state=hf_state
        )
        dexcg.name = "DexcG"

        pcu2 = build_particle_conserving_u2(
            num_spatial_orbitals=n_orbitals,
            num_particles=particles,
            reps=2,
            qubit_mapper=mapper
        )

        uccsd = UCC(
            num_spatial_orbitals=n_orbitals,
            num_particles=particles,
            excitations="sd",
            qubit_mapper=mapper,
            initial_state=hf_state
        )
        uccsd.name = "UCCSD"

        raw_dict = {"DexcG": dexcg, "PCU2": pcu2, "UCCSD": uccsd}
    else:
        raw_dict = get_ansatz_dict(num_spatial_orbitals=n_orbitals, num_particles=particles)

    # Pre-decompose circuits
    ansatze_decomp = {}
    for name, qc in raw_dict.items():
        dec = qc.decompose()
        if any(op.name in ["PauliEvolution", "HartreeFock"] for op in dec.data):
            dec = dec.decompose()
        ansatze_decomp[name] = dec

    return ansatze_decomp


def run_4e4o_benchmark(force: bool = False):
    """Executes or loads 4e4o (8 qubits) benchmark grid."""
    print("=" * 74)
    print("RUNNING 4e4o (8 QUBITS) VQE BENCHMARK GRID")
    print("=" * 74)
    cache_path = DATA_DIR / "benchmark_results_6-31gd_p_4e4o.json"

    if cache_path.exists() and not force:
        print(f"  -> Loading cached 4e4o benchmark results from {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rdf_4 = pd.DataFrame(data["results"])
        cdf_4 = pd.DataFrame(data["convergence"])
        _print_summary(rdf_4, "4e4o")
        return rdf_4, cdf_4, None

    t0 = time.time()
    rdf_4, cdf_4, meta_4 = run_full_vqe_benchmark(
        basis="6-31g(d,p)",
        n_active_electrons=4,
        n_active_orbitals=4,
        maxiter=25,
        verbose=True,
        use_cache=False
    )
    print(f"\n[DONE] 4e4o benchmark completed in {time.time() - t0:.2f} s")
    _print_summary(rdf_4, "4e4o")
    return rdf_4, cdf_4, meta_4


def run_6e6o_benchmark(force: bool = False, exhaustive: bool = False):
    """
    Executes or loads 6e6o (12 qubits) benchmark grid strictly serially.
    Skips k-UpCCGSD and restricts to SPSA by default.
    """
    print("\n" + "=" * 74)
    print("RUNNING 6e6o (12 QUBITS) VQE BENCHMARK")
    print("=" * 74)
    cache_path = DATA_DIR / "benchmark_results_6-31gd_p_6e6o.json"

    if cache_path.exists() and not force:
        print(f"  -> Loading cached 6e6o benchmark results from {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rdf_6 = pd.DataFrame(data["results"])
        cdf_6 = pd.DataFrame(data["convergence"])
        _print_summary(rdf_6, "6e6o")
        return rdf_6, cdf_6, None

    t0 = time.time()
    H, exact_energy, meta = load_or_build_bn_dot_hamiltonian(
        basis="6-31g(d,p)",
        n_active_electrons=6,
        n_active_orbitals=6
    )
    hf_energy = meta["hf_energy"]
    ref_det_energy = meta.get("reference_determinant_energy", hf_energy)
    delta_corr = abs(ref_det_energy - exact_energy)
    E_corr_mHa = meta["correlation_energy_mHa"]

    print(f"  Qubits: {H.num_qubits} | Pauli Terms: {len(H)}")
    print(f"  Reference Determinant: {ref_det_energy:.8f} Ha | Exact CASCI: {exact_energy:.8f} Ha")
    print(f"  Correlation Energy E_corr: {E_corr_mHa:.5f} mHa")
    if E_corr_mHa < 1.6:
        print(f"  >>> [WARNING] 6e6o Correlation Energy ({E_corr_mHa:.4f} mHa) < Chemical Accuracy (1.6 mHa)! <<<")

    ansatze_decomp = get_active_space_ansatze(6, 6)
    estimator = StatevectorEstimator()
    sampler = StatevectorSampler()

    inits = ["zero", "half", "one", "random"]
    all_ansatze = ["DexcG", "PCU2", "UCCSD", "k-UpCCGSD"]

    # Calculate total runs
    plan = []
    for a_name in all_ansatze:
        if a_name == "k-UpCCGSD":
            continue
        opts = ["SPSA"] if not exhaustive else ["SPSA", "GD", "ADAM"]
        for i_name in inits:
            for o_name in opts:
                iters = 15 if o_name == "SPSA" else 10
                plan.append((a_name, i_name, o_name, iters))

    print(f"\n  Plan: {len(plan)} configurations scheduled (strictly serial).")
    results = []
    convergence_data = {}

    with tqdm(total=len(plan), desc="6e6o Benchmarking") as pbar:
        for a_name in all_ansatze:
            if a_name == "k-UpCCGSD":
                pbar.write(f"WARNING: Skipping {a_name} at (6, 6) to prevent parameter-binding deadlock.")
                continue

            qc = ansatze_decomp[a_name]
            opts = ["SPSA"] if not exhaustive else ["SPSA", "GD", "ADAM"]

            for i_name in inits:
                for o_name in opts:
                    iters = 15 if o_name == "SPSA" else 10
                    config_str = f"Space=6e6o, Ansatz={a_name}, Init={i_name}, Opt={o_name}"
                    pbar.set_postfix_str(config_str)
                    pbar.write(f"Evaluating: {config_str} ({iters} iter, {qc.num_parameters} params)...")

                    t_run0 = time.perf_counter()
                    res = run_vqe_single(
                        circuit=qc,
                        hamiltonian=H,
                        ansatz_name=a_name,
                        init_name=i_name,
                        optimizer_name=o_name,
                        maxiter=iters,
                        estimator=estimator,
                        sampler=sampler,
                        seed=42
                    )
                    wtime = time.perf_counter() - t_run0

                    final_e = res["final_energy"]
                    err_mHa = abs(final_e - exact_energy) * 1000.0
                    pct_corr = ((ref_det_energy - final_e) / delta_corr) if delta_corr > 1e-12 else 1.0

                    run_idx = len(results) + 1
                    results.append({
                        "Config_ID": run_idx,
                        "Ansatz": a_name,
                        "Initialization": i_name,
                        "Optimizer": o_name,
                        "Parameters": res["num_params"],
                        "Final_Energy_Ha": final_e,
                        "Best_Energy_Ha": res["best_energy"],
                        "Exact_Energy_Ha": exact_energy,
                        "Error_mHa": err_mHa,
                        "Pct_Corr_Recovered": pct_corr,
                        "Rel_Error_Pct": (abs(final_e - exact_energy) / abs(exact_energy)) * 100.0,
                        "Total_Evaluations": res["total_evaluations"],
                        "Wall_Time_s": wtime,
                        "Iterations": iters
                    })

                    cfg_label = f"{a_name}_{i_name}_{o_name}"
                    convergence_data[cfg_label] = res["energy_history"]

                    pbar.write(f"  -> Result: E={final_e:.8f} Ha | Err={err_mHa:8.4f} mHa | %Corr={pct_corr*100:6.2f}% | Time={wtime:.2f}s")
                    pbar.update(1)

    rdf_6 = pd.DataFrame(results)

    actual_len = min(len(hist) for hist in convergence_data.values()) if convergence_data else 16
    conv_records = []
    for it in range(actual_len):
        row = {"Iteration": it}
        for cfg, hist in convergence_data.items():
            row[cfg] = hist[it] if it < len(hist) else hist[-1]
        conv_records.append(row)
    cdf_6 = pd.DataFrame(conv_records)

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({
            "results": rdf_6.to_dict(orient="records"),
            "convergence": cdf_6.to_dict(orient="records")
        }, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n[DONE] 6e6o benchmark completed in {elapsed:.2f} s ({elapsed/60:.1f} min)")
    print(f"Saved cache to {cache_path}")
    _print_summary(rdf_6, "6e6o")

    return rdf_6, cdf_6, meta


def update_results_workbook():
    """Regenerates results.xlsx workbook with all sheets (2e2o, 4e4o, 6e6o, scaling)."""
    from src.excel_export import export_benchmark_to_excel
    from src.scaling import get_qubit_and_parameter_scaling_table
    from src.ansatze import analyze_and_render_circuits
    from tests.run_verification_suite import run_all_verifications

    print("\n" + "=" * 74)
    print("UPDATING results.xlsx WITH ALL ACTIVE-SPACE DATA")
    print("=" * 74)

    # 1. Load 2e2o
    with open(DATA_DIR / "benchmark_results_6-31gd_p.json", "r", encoding="utf-8") as f:
        b2 = json.load(f)
    rdf_2 = pd.DataFrame(b2["results"])
    cdf_2 = pd.DataFrame(b2["convergence"])

    with open(DATA_DIR / "bn_dot_631gdp.json", "r", encoding="utf-8") as f:
        meta_2 = json.load(f)

    # 2. Load 4e4o
    p4 = DATA_DIR / "benchmark_results_6-31gd_p_4e4o.json"
    rdf_4 = None
    if p4.exists():
        with open(p4, "r", encoding="utf-8") as f:
            b4 = json.load(f)
        rdf_4 = pd.DataFrame(b4["results"])
        print(f"  Loaded 4e4o results: {len(rdf_4)} rows")

    # 3. Load 6e6o
    p6 = DATA_DIR / "benchmark_results_6-31gd_p_6e6o.json"
    rdf_6 = None
    if p6.exists():
        with open(p6, "r", encoding="utf-8") as f:
            b6 = json.load(f)
        rdf_6 = pd.DataFrame(b6["results"])
        print(f"  Loaded 6e6o results: {len(rdf_6)} rows")

    # 4. Analytical scaling table
    scaling_df = get_qubit_and_parameter_scaling_table()

    # 5. Circuit metrics (2e2o)
    ansatze_2e2o = get_ansatz_dict(num_spatial_orbitals=2, num_particles=(1, 1))
    circuits_df = analyze_and_render_circuits(ansatze_2e2o, save_pngs=False)

    # 6. LR sweep
    lr_sweep_df = None
    p_lr = DATA_DIR / "lr_sweep_results.json"
    if p_lr.exists():
        with open(p_lr, "r", encoding="utf-8") as f:
            lr_sweep_df = pd.DataFrame(json.load(f))

    # 7. Robustness
    robustness_df = None
    p_rob = DATA_DIR / "robustness_results.json"
    if p_rob.exists():
        with open(p_rob, "r", encoding="utf-8") as f:
            robustness_df = pd.DataFrame(json.load(f))

    # 8. Physical Hardware Run
    hw_data_for_sheet = None
    p_hw = DATA_DIR / "hardware_run.json"
    if p_hw.exists():
        with open(p_hw, "r", encoding="utf-8") as f:
            hw_record = json.load(f)
        hw_data_for_sheet = {
            "Status": f"Evaluated on Physical IBM Quantum QPU ({hw_record.get('backend')})",
            "Target Backend": hw_record.get("backend", "ibm_fez"),
            "Job ID": hw_record.get("job_id"),
            "Ansatz Selected": f"{hw_record.get('ansatz', 'DexcG')} (theta sweep [-0.10, +0.10])",
            "Parameters Optimized": 1,
            "Transpiled 2-Qubit Gate Count": hw_record.get("transpiled_2q_gates", 42),
            "Circuit Depth": hw_record.get("transpiled_depth", 144),
            "Exact Active Ground Energy (Ha)": meta_2["casci_energy"],
            "Measured Energy (Ha)": hw_record.get("measured_energy_ha"),
            "Energy Error (mHa)": hw_record.get("error_mha"),
            "Shots": hw_record.get("shots", 4096),
            "QPU Runtime (seconds)": hw_record.get("qpu_seconds"),
            "Direct Job Dashboard": f"https://quantum.ibm.com/jobs/{hw_record.get('job_id')}"
        }

    # 9. Parameter Binding Test
    from src.hardware import test_transpilation_parameter_binding
    best_ansatz_qc = ansatze_2e2o["DexcG"]
    binding_test_data = test_transpilation_parameter_binding(best_ansatz_qc, backend_name="ibm_fez")

    # 10. Automated Verifications
    verification_df = run_all_verifications()

    # Export
    export_benchmark_to_excel(
        results_df=rdf_2,
        convergence_df=cdf_2,
        circuits_df=circuits_df,
        meta=meta_2,
        lr_sweep_df=lr_sweep_df,
        robustness_df=robustness_df,
        hardware_data=hw_data_for_sheet,
        binding_test_data=binding_test_data,
        verification_df=verification_df,
        results_4e4o_df=rdf_4,
        results_6e6o_df=rdf_6,
        scaling_df=scaling_df,
        output_path="results.xlsx"
    )
    print("  -> results.xlsx updated successfully with all active space sheets!")


def _print_summary(rdf: pd.DataFrame, label: str):
    chem_acc = rdf[rdf["Error_mHa"] < 1.6]
    print(f"\n  {label} chemical accuracy (< 1.6 mHa): {len(chem_acc)} / {len(rdf)} ({len(chem_acc)/len(rdf)*100:.1f}%)")
    cols = ["Ansatz", "Initialization", "Optimizer", "Final_Energy_Ha", "Error_mHa", "Pct_Corr_Recovered", "Total_Evaluations", "Wall_Time_s"]
    available = [c for c in cols if c in rdf.columns]
    print(f"  Top 10 configurations:")
    print(rdf.sort_values(by=["Error_mHa", "Total_Evaluations"]).head(10)[available].to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-active-space serial VQE benchmarks for B8N8H10")
    parser.add_argument("--skip-6e6o", action="store_true", help="Skip 12-qubit (6e6o) active space entirely")
    parser.add_argument("--skip-excel", action="store_true", help="Skip Excel export")
    parser.add_argument("--force-4e4o", action="store_true", help="Recompute 4e4o even if cache exists")
    parser.add_argument("--force-6e6o", action="store_true", help="Recompute 6e6o even if cache exists")
    parser.add_argument("--exhaustive-6e6o", action="store_true", help="Run GD/ADAM on 6e6o (default: SPSA-only)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("==========================================================================")
    print("STRICTLY SERIAL MULTI-ACTIVE-SPACE VQE BENCHMARK")
    print("==========================================================================")
    print(f"Skip 6e6o: {args.skip_6e6o}")
    print(f"Force 4e4o: {args.force_4e4o}")
    print(f"Force 6e6o: {args.force_6e6o}")
    print(f"Exhaustive 6e6o: {args.exhaustive_6e6o}")
    print()

    # 1. 4e4o space
    rdf_4, cdf_4, meta_4 = run_4e4o_benchmark(force=args.force_4e4o)

    # 2. 6e6o space
    if not args.skip_6e6o:
        rdf_6, cdf_6, meta_6 = run_6e6o_benchmark(
            force=args.force_6e6o,
            exhaustive=args.exhaustive_6e6o
        )

    # 3. Workbook generation
    if not args.skip_excel:
        update_results_workbook()

