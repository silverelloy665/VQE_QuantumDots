"""
Benchmark runner for the 64-run VQE grid on BN Quantum Dot (B8N8H10).
Includes learning-rate sweep, 5-seed random init robustness analysis,
and explicit tie ranking.
"""
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import SparseEfficiencyWarning
from qiskit.primitives import StatevectorEstimator, StatevectorSampler

from src.molecule import load_or_build_bn_dot_hamiltonian
from src.ansatze import get_ansatz_dict
from src.optimizers import run_vqe_single

warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def run_lr_sweep(
    basis: str = "6-31g(d,p)",
    lr_candidates: list[float] | None = None,
    sweep_seed: int = 123,
    maxiter: int = 25,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Executes a learning rate tuning sweep across candidate learning rates
    on a held-out seed to determine optimal learning rates per optimizer.
    """
    if lr_candidates is None:
        lr_candidates = [0.005, 0.01, 0.02, 0.05, 0.1, 0.5]
        
    cache_path = DATA_DIR / "lr_sweep_results.json"
    if use_cache and cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)
        
    hamiltonian, exact_energy, meta = load_or_build_bn_dot_hamiltonian(basis=basis)
    ansatze = get_ansatz_dict()
    estimator = StatevectorEstimator()
    sampler = StatevectorSampler()
    
    sweep_records = []
    # Test on representative ansatz: DexcG from half initialization
    qc = ansatze["DexcG"]
    
    for opt in ["GD", "ADAM", "SPSA", "QNSPSA"]:
        for lr in lr_candidates:
            res = run_vqe_single(
                circuit=qc,
                hamiltonian=hamiltonian,
                ansatz_name="DexcG",
                init_name="half",
                optimizer_name=opt,
                maxiter=maxiter,
                lr=lr,
                estimator=estimator,
                sampler=sampler,
                seed=sweep_seed
            )
            err_mHa = abs(res["final_energy"] - exact_energy) * 1000.0
            sweep_records.append({
                "Optimizer": opt,
                "Learning_Rate": lr,
                "Final_Energy_Ha": res["final_energy"],
                "Error_mHa": err_mHa,
                "Total_Evaluations": res["total_evaluations"],
                "Wall_Time_s": res["wall_time"]
            })
            
    df_sweep = pd.DataFrame(sweep_records)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(df_sweep.to_dict(orient="records"), f, indent=2)
        
    return df_sweep

def run_robustness_benchmark(
    basis: str = "6-31g(d,p)",
    seeds: list[int] | None = None,
    maxiter: int = 50,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Evaluates random initialization robustness across 5 fixed random seeds.
    Computes mean, standard deviation, minimum, and maximum error for each Ansatz x Optimizer.
    """
    if seeds is None:
        seeds = [42, 123, 456, 789, 1000]
        
    cache_path = DATA_DIR / "robustness_results.json"
    if use_cache and cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)
        
    hamiltonian, exact_energy, meta = load_or_build_bn_dot_hamiltonian(basis=basis)
    ref_det_energy = meta.get("reference_determinant_energy", meta["hf_energy"])
    delta_corr = abs(ref_det_energy - exact_energy)
    
    ansatze = get_ansatz_dict()
    estimator = StatevectorEstimator()
    sampler = StatevectorSampler()
    
    records = []
    ansatz_names = ["DexcG", "PCU2", "UCCSD", "k-UpCCGSD"]
    optimizer_names = ["GD", "ADAM", "SPSA", "QNSPSA"]
    
    for a_name in ansatz_names:
        qc = ansatze[a_name]
        for o_name in optimizer_names:
            errors = []
            pct_corrs = []
            for s in seeds:
                res = run_vqe_single(
                    circuit=qc,
                    hamiltonian=hamiltonian,
                    ansatz_name=a_name,
                    init_name="random",
                    optimizer_name=o_name,
                    maxiter=maxiter,
                    estimator=estimator,
                    sampler=sampler,
                    seed=s
                )
                err = abs(res["final_energy"] - exact_energy) * 1000.0
                pct = ((ref_det_energy - res["final_energy"]) / delta_corr) * 100.0 if delta_corr > 1e-12 else 100.0
                errors.append(err)
                pct_corrs.append(pct)
                
            records.append({
                "Ansatz": a_name,
                "Optimizer": o_name,
                "Seeds_Count": len(seeds),
                "Mean_Error_mHa": float(np.mean(errors)),
                "Std_Error_mHa": float(np.std(errors)),
                "Min_Error_mHa": float(np.min(errors)),
                "Max_Error_mHa": float(np.max(errors)),
                "Mean_Pct_Corr": float(np.mean(pct_corrs))
            })
            
    df_rob = pd.DataFrame(records)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(df_rob.to_dict(orient="records"), f, indent=2)
        
    return df_rob

def run_full_vqe_benchmark(
    basis: str = "6-31g(d,p)",
    n_active_electrons: int = 2,
    n_active_orbitals: int = 2,
    orbital_method: str = "rhf",
    maxiter: int = 50,
    seed: int = 42,
    verbose: bool = True,
    use_cache: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Executes the full 64-run VQE benchmark grid: 4 ansätze x 4 inits x 4 optimizers.
    Calculates % correlation recovered, total evaluations, best vs last energy, and explicit tie ranking.
    """
    hamiltonian, exact_energy, meta = load_or_build_bn_dot_hamiltonian(
        basis=basis,
        n_active_electrons=n_active_electrons,
        n_active_orbitals=n_active_orbitals,
        orbital_method=orbital_method
    )
    
    hf_energy = meta["hf_energy"]
    ref_det_energy = meta.get("reference_determinant_energy", hf_energy)
    delta_corr = abs(ref_det_energy - exact_energy)
    
    clean_basis = "6-31gd_p" if "6-31g" in basis.lower() else "sto3g"
    if (n_active_electrons, n_active_orbitals) == (4, 4):
        cache_path = DATA_DIR / f"benchmark_results_{clean_basis}_4e4o.json"
    elif (n_active_electrons, n_active_orbitals) == (6, 6):
        cache_path = DATA_DIR / f"benchmark_results_{clean_basis}_6e6o.json"
    else:
        cache_path = DATA_DIR / f"benchmark_results_{clean_basis}.json"

    if use_cache and cache_path.exists():
        if verbose:
            print(f"[CACHE] Loading cached benchmark results from {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        results_df = pd.DataFrame(data["results"])
        convergence_df = pd.DataFrame(data["convergence"])
        return results_df, convergence_df, meta
    
    ansatze = get_ansatz_dict(
        num_spatial_orbitals=n_active_orbitals,
        num_particles=(n_active_electrons // 2, n_active_electrons // 2)
    )
    
    # Pre-decompose UCC circuits to eliminate repetitive runtime gate synthesis overhead
    ansatze_decomp = {}
    for a_name, qc in ansatze.items():
        d = qc.decompose()
        if any(op.name in ["PauliEvolution", "HartreeFock"] for op in d.data):
            d = d.decompose()
        ansatze_decomp[a_name] = d

    ansatz_names = ["DexcG", "PCU2", "UCCSD", "k-UpCCGSD"]
    init_names = ["zero", "half", "one", "random"]
    optimizer_names = ["GD", "ADAM", "SPSA", "QNSPSA"]
    
    estimator = StatevectorEstimator()
    try:
        from qiskit_aer.primitives import EstimatorV2 as AerEstimator
        estimator = AerEstimator()
    except Exception:
        estimator = StatevectorEstimator()
    sampler = StatevectorSampler()
    
    results = []
    convergence_data = {}
    
    total_runs = len(ansatz_names) * len(init_names) * len(optimizer_names)
    run_idx = 0
    
    is_smoke_test = (n_active_electrons, n_active_orbitals) == (2, 2)
    tag = " [SMOKE TEST / TRIVIAL ACTIVE SPACE: E_corr = 0.041 mHa << 1.6 mHa]" if is_smoke_test else ""
    
    if verbose:
        print("==========================================================================")
        print(f"Executing VQE Benchmark for {meta['label']} ({meta['molecule']}) | Basis: {basis}{tag}")
        print(f"Active Space: ({n_active_electrons}e, {n_active_orbitals}o) -> {meta['num_qubits']} qubits | Pauli terms: {meta['num_pauli_terms']}")
        print(f"Total Configurations: {total_runs} (4 ansatz x 4 inits x 4 optimizers, {maxiter} iter each)")
        print(f"Ref Det Reference: {ref_det_energy:.8f} Ha | Exact CASCI: {exact_energy:.8f} Ha | E_corr: {meta['correlation_energy_mHa']:.5f} mHa")
        print(f"Chemical Accuracy Threshold: 1.6000 mHa (HF baseline is {meta['correlation_energy_mHa']:.4f} mHa from exact)")
        print("==========================================================================")

    for a_name in ansatz_names:
        qc = ansatze_decomp[a_name]
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
                best_e = res["best_energy"]
                error_mHa = abs(final_e - exact_energy) * 1000.0
                rel_error = (abs(final_e - exact_energy) / abs(exact_energy)) * 100.0
                
                pct_corr = ((ref_det_energy - final_e) / delta_corr) if delta_corr > 1e-12 else 1.0
                
                results.append({
                    "Config_ID": run_idx,
                    "Ansatz": a_name,
                    "Initialization": i_name,
                    "Optimizer": o_name,
                    "Parameters": res["num_params"],
                    "Final_Energy_Ha": final_e,
                    "Best_Energy_Ha": best_e,
                    "Exact_Energy_Ha": exact_energy,
                    "Error_mHa": error_mHa,
                    "Pct_Corr_Recovered": pct_corr,
                    "Rel_Error_Pct": rel_error,
                    "Total_Evaluations": res["total_evaluations"],
                    "Wall_Time_s": res["wall_time"],
                    "Iterations": maxiter
                })
                
                convergence_data[cfg_label] = res["energy_history"]
                
                if verbose:
                    print(f"[{run_idx:02d}/{total_runs}] {a_name:10s} | {i_name:6s} | {o_name:6s} -> "
                          f"E_final: {final_e:.8f} Ha | Err: {error_mHa:8.4f} mHa | %Corr: {pct_corr*100:6.2f}% | Evals: {res['total_evaluations']:4d} | Time: {res['wall_time']:.3f}s")
                    
    results_df = pd.DataFrame(results)
    
    # Build convergence DataFrame: use the shortest history length across all configs
    # (handles variable-length histories from different optimizers/maxiter settings)
    conv_records = []
    if convergence_data:
        actual_len = min(len(hist) for hist in convergence_data.values())
    else:
        actual_len = maxiter + 1
        
    for it_idx in range(actual_len):
        row = {"Iteration": it_idx}
        for cfg, hist in convergence_data.items():
            row[cfg] = hist[it_idx] if it_idx < len(hist) else hist[-1]
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
        print("==========================================================================")
        print("VQE Benchmark Grid Completed Successfully!")
        print("==========================================================================")
        
    return results_df, convergence_df, meta


if __name__ == "__main__":
    rdf, cdf, meta = run_full_vqe_benchmark(basis="6-31g(d,p)", maxiter=50)
    print("\nTop Configurations (Tie Ranking):")
    print(rdf.sort_values(by=["Error_mHa", "Total_Evaluations", "Wall_Time_s"]).head(10)[["Ansatz", "Initialization", "Optimizer", "Final_Energy_Ha", "Error_mHa", "Pct_Corr_Recovered", "Total_Evaluations", "Wall_Time_s"]].to_string(index=False))