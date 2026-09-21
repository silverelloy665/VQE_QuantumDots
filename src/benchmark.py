"""
Benchmark runner for the 64-run VQE grid on BN Quantum Dot (B8N8H10).
"""
import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy.sparse import SparseEfficiencyWarning
from qiskit.primitives import StatevectorEstimator, StatevectorSampler

from src.molecule import load_or_build_bn_dot_hamiltonian
from src.ansatze import get_ansatz_dict
from src.optimizers import run_vqe_single

warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def run_full_vqe_benchmark(
    basis: str = "6-31g(d,p)",
    maxiter: int = 50,
    seed: int = 42,
    verbose: bool = True,
    use_cache: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Executes all 64 VQE runs: 4 ansätze x 4 inits x 4 optimizers (50 iterations each).
    
    Returns:
        results_df (pd.DataFrame): 64 rows with metrics.
        convergence_df (pd.DataFrame): Iteration-by-iteration energies.
        meta (dict): Experiment metadata and exact energy reference.
    """
    hamiltonian, exact_energy, meta = load_or_build_bn_dot_hamiltonian(basis=basis)
    ansatze = get_ansatz_dict()
    
    cache_path = DATA_DIR / f"benchmark_results_{basis.replace('(', '').replace(')', '').replace(',', '_')}.json"
    if use_cache and cache_path.exists():
        if verbose:
            print(f"[CACHE] Loading cached benchmark results from {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        results_df = pd.DataFrame(data["results"])
        convergence_df = pd.DataFrame(data["convergence"])
        return results_df, convergence_df, meta
    
    ansatz_names = ["DexcG", "PCU2", "UCCSD", "k-UpCCGSD"]
    init_names = ["zero", "half", "one", "random"]
    optimizer_names = ["GD", "ADAM", "SPSA", "QNSPSA"]
    
    estimator = StatevectorEstimator()
    sampler = StatevectorSampler()
    
    results = []
    convergence_data = {}
    
    total_runs = len(ansatz_names) * len(init_names) * len(optimizer_names)
    run_idx = 0
    
    if verbose:
        print(f"==========================================================================")
        print(f"Starting VQE Benchmark for {meta['label']} ({meta['molecule']}) | Basis: {basis}")
        print(f"Total Configurations: {total_runs} (4 ansatz x 4 inits x 4 optimizers, {maxiter} iter each)")
        print(f"Exact Active-Space Reference Ground Energy: {exact_energy:.8f} Ha")
        print(f"==========================================================================")

    for a_name in ansatz_names:
        qc = ansatze[a_name]
        for i_name in init_names:
            for o_name in optimizer_names:
                run_idx += 1
                cfg_label = f"{a_name}_{i_name}_{o_name}"
                
                res = run_vqe_single(
                    circuit=qc,
                    hamiltonian=hamiltonian,
                    ansatz_name=a_name,
                    init_name=i_name,
                    optimizer_name=o_name,
                    maxiter=maxiter,
                    estimator=estimator,
                    sampler=sampler,
                    seed=seed
                )
                
                final_e = res["final_energy"]
                error_mHa = abs(final_e - exact_energy) * 1000.0
                rel_error = (abs(final_e - exact_energy) / abs(exact_energy)) * 100.0
                
                results.append({
                    "Config_ID": run_idx,
                    "Ansatz": a_name,
                    "Initialization": i_name,
                    "Optimizer": o_name,
                    "Parameters": res["num_params"],
                    "Final_Energy_Ha": final_e,
                    "Exact_Energy_Ha": exact_energy,
                    "Error_mHa": error_mHa,
                    "Rel_Error_Pct": rel_error,
                    "Wall_Time_s": res["wall_time"],
                    "Iterations": maxiter
                })
                
                convergence_data[cfg_label] = res["energy_history"]
                
                if verbose:
                    print(f"[{run_idx:02d}/{total_runs}] {a_name:10s} | {i_name:6s} | {o_name:6s} -> "
                          f"E_final: {final_e:.8f} Ha | Err: {error_mHa:8.4f} mHa | Time: {res['wall_time']:.3f}s")
                    
    results_df = pd.DataFrame(results)
    
    # Create convergence DataFrame with iterations as rows
    conv_records = []
    for it in range(maxiter):
        row = {"Iteration": it + 1}
        for cfg, hist in convergence_data.items():
            row[cfg] = hist[it] if it < len(hist) else hist[-1]
        conv_records.append(row)
    convergence_df = pd.DataFrame(conv_records)
    
    # Save cache
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({
            "results": results_df.to_dict(orient="records"),
            "convergence": convergence_df.to_dict(orient="records")
        }, f, indent=2)

    if verbose:
        print(f"==========================================================================")
        print("VQE Benchmark Grid Completed Successfully!")
        print(f"==========================================================================")
        
    return results_df, convergence_df, meta

if __name__ == "__main__":
    rdf, cdf, meta = run_full_vqe_benchmark(basis="6-31g(d,p)", maxiter=50)
    print("\nTop 5 Best Configurations:")
    print(rdf.sort_values(by="Error_mHa").head(5)[["Ansatz", "Initialization", "Optimizer", "Final_Energy_Ha", "Error_mHa", "Wall_Time_s"]].to_string(index=False))

