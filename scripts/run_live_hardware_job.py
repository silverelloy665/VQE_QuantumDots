"""
Batch hardware job preparation script for IBM Quantum hardware (ibm_marrakesh).
Prepares and transpiles circuits with error-mitigation configurations (DD, Pauli Twirling, ZNE)
and generates the job specification without submitting to physical QPU.
"""
import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from qiskit.circuit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.providers.fake_provider import GenericBackendV2

from src.molecule import load_or_build_bn_dot_hamiltonian
from src.ansatze import get_ansatz_dict
from src.optimizers import run_rotosolve_dexcg, run_lbfgsb
from src.config import get_runtime_service, DEFAULT_HARDWARE_BACKEND

DATA_DIR = REPO_ROOT / "data"


def get_optimal_parameters(ansatze, hamiltonian, exact_energy):
    """
    Loads optimal parameters from data/benchmark_results_6-31gd_p.json if available.
    Otherwise computes them directly via fast exact optimizers (Rotosolve / L-BFGS-B).
    """
    cache_path = DATA_DIR / "benchmark_results_6-31gd_p.json"
    dexcg_params = None
    uccsd_params = None
    
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cdata = json.load(f)
            records = cdata.get("results", [])
            dexcg_records = [r for r in records if r["Ansatz"] == "DexcG" and "Best_Params" in r]
            if dexcg_records:
                best_d = min(dexcg_records, key=lambda r: r["Error_mHa"])
                dexcg_params = np.array(best_d["Best_Params"])
                
            uccsd_records = [r for r in records if r["Ansatz"] == "UCCSD" and "Best_Params" in r]
            if uccsd_records:
                best_u = min(uccsd_records, key=lambda r: r["Error_mHa"])
                uccsd_params = np.array(best_u["Best_Params"])
        except Exception as e:
            print(f"[Warning] Failed to parse cache {cache_path}: {e}")

    # Fallback to fast exact statevector calculation if not cached
    if dexcg_params is None:
        print("[Optimization] Computing optimal theta* for DexcG via Rotosolve...")
        _, p_star = run_rotosolve_dexcg(ansatze["DexcG"], hamiltonian)
        dexcg_params = p_star
        
    if uccsd_params is None:
        print("[Optimization] Computing optimal theta* for UCCSD via L-BFGS-B...")
        _, p_star, _, _, _ = run_lbfgsb(ansatze["UCCSD"], hamiltonian, np.zeros(ansatze["UCCSD"].num_parameters), maxiter=25)
        uccsd_params = p_star
        
    return dexcg_params, uccsd_params


def prepare_hardware_job_specs(
    backend_name: str = DEFAULT_HARDWARE_BACKEND,
    shots: int = 8192,
    seed_transpiler: int = 42
) -> pd.DataFrame:
    """
    Prepares 3 circuits (HF baseline, DexcG theta*, UCCSD theta*),
    transpiles them to the target hardware layout, applies 3 error-mitigation
    configurations (Dynamical Decoupling, Pauli Twirling, ZNE),
    and exports data/hardware_job_spec.json. Does NOT submit physical jobs.
    """
    hamiltonian, exact_energy, meta = load_or_build_bn_dot_hamiltonian("6-31g(d,p)")
    ansatze = get_ansatz_dict()
    
    # Resolve backend
    backend = None
    try:
        service = get_runtime_service()
        backend = service.backend(backend_name)
        print(f"[Backend] Successfully resolved live hardware target: {backend.name}")
    except Exception as e:
        print(f"[Backend Warning] Could not connect to live backend {backend_name} ({e}). Using GenericBackendV2 mock.")
        backend = GenericBackendV2(num_qubits=156, basis_gates=["ecr", "id", "rz", "sx", "x"])

    dexcg_params, uccsd_params = get_optimal_parameters(ansatze, hamiltonian, exact_energy)
    
    # Define 3 circuits to benchmark
    # 1. HF baseline (theta = 0)
    hf_qc = ansatze["DexcG"].assign_parameters(np.zeros(ansatze["DexcG"].num_parameters))
    hf_qc.name = "HF_baseline"
    
    # 2. DexcG optimal
    dexcg_qc = ansatze["DexcG"].assign_parameters(dexcg_params)
    dexcg_qc.name = "DexcG_optimal"
    
    # 3. UCCSD optimal
    uccsd_qc = ansatze["UCCSD"].assign_parameters(uccsd_params)
    uccsd_qc.name = "UCCSD_optimal"
    
    circuits = [
        ("HF_baseline", hf_qc, np.zeros(ansatze["DexcG"].num_parameters)),
        ("DexcG_optimal", dexcg_qc, dexcg_params),
        ("UCCSD_optimal", uccsd_qc, uccsd_params)
    ]
    
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3, seed_transpiler=seed_transpiler)
    
    resilience_configs = [
        ("dynamical_decoupling", {"dynamical_decoupling": {"enable": True, "sequence_type": "XX"}}),
        ("pauli_twirling", {"twirling": {"enable_gates": True, "enable_measure": True}}),
        ("zne", {"resilience": {"measure_mitigation": True, "zne_mitigation": True, "zne": {"extrapolator": "exponential"}}})
    ]
    
    job_specs = []
    summary_rows = []
    
    for c_name, qc, p_vals in circuits:
        decomposed = qc.decompose()
        if any(op.name in ["PauliEvolution", "HartreeFock"] for op in decomposed.data):
            decomposed = decomposed.decompose()
            
        transpiled = pm.run(decomposed)
        t_ops = transpiled.count_ops()
        n_2q = sum(count for op, count in t_ops.items() if op in ["cx", "ecr", "cz"])
        depth = transpiled.depth()
        
        for r_name, r_opts in resilience_configs:
            spec = {
                "circuit_name": c_name,
                "backend": getattr(backend, "name", backend_name),
                "shots": shots,
                "resilience_type": r_name,
                "optimization_level": 3,
                "seed_transpiler": seed_transpiler,
                "2Q_gates": n_2q,
                "depth": depth,
                "parameters": [float(x) for x in p_vals],
                "options": r_opts,
                "status": "PREPARED (NOT SUBMITTED)"
            }
            job_specs.append(spec)
            summary_rows.append({
                "Circuit": c_name,
                "Backend": getattr(backend, "name", backend_name),
                "Shots": shots,
                "Resilience": r_name,
                "2Q_Gates": n_2q,
                "Depth": depth,
                "Status": "PREPARED"
            })
            
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "hardware_job_spec.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(job_specs, f, indent=2)
        
    print(f"\nSaved batch hardware job specification to {out_path} ({len(job_specs)} jobs, NOT submitted).")
    
    df_summary = pd.DataFrame(summary_rows)
    return df_summary


if __name__ == "__main__":
    print("=" * 70)
    print("Preparing IBM Quantum Batch Hardware Job Specification")
    print("=" * 70)
    df = prepare_hardware_job_specs()
    print("\nBatch Hardware Job Spec Summary:")
    print(df.to_string(index=False))
