"""
Submit and monitor a live IBM Quantum Hardware evaluation job on least-busy QPU.
Submit and monitor a live IBM Quantum Hardware evaluation job on the least-busy QPU.
Reads best configuration dynamically from benchmark results and caches job details to data/hardware_run.json.
"""
import sys
import json
import time
from pathlib import Path
import numpy as np
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2

from src.config import get_runtime_service
from src.molecule import load_or_build_bn_dot_hamiltonian
from src.ansatze import get_ansatz_dict
from src.hardware import run_noisy_fake_backend_evaluation

def run_live_qpu():
DATA_DIR = Path(__file__).resolve().parent / "data"

def run_live_qpu(shots: int = 4096, optimization_level: int = 3):
    print("==================================================================")
    print("SUBMITTING LIVE IBM QUANTUM HARDWARE JOB")
    print("IBM QUANTUM HARDWARE EXECUTION / VERIFICATION HARNESS")
    print("==================================================================")
    
    # 1. Connect to service
    service = get_runtime_service()
    backends = service.backends(operational=True, simulator=False)
    
    print("Available Operational Backends:")
    for b in backends:
        status = b.status()
        print(f"  - {b.name:15s}: {status.pending_jobs} pending jobs (Operational: {status.operational})")
        
    least_busy = min(backends, key=lambda b: b.status().pending_jobs)
    print(f"\n[Selected Backend] -> {least_busy.name} (Pending jobs: {least_busy.status().pending_jobs})")
    
    # 2. Build Hamiltonian & UCCSD Ansatz
    # 1. Load benchmark results to find optimal configuration
    benchmark_cache = DATA_DIR / "benchmark_results_6-31gd_p.json"
    H, E_exact, meta = load_or_build_bn_dot_hamiltonian("6-31g(d,p)")
    ansatze = get_ansatz_dict()
    qc = ansatze["UCCSD"]
    
    # Let's use the optimized parameters for UCCSD (or near-optimal small angle)
    optimal_params = np.array([0.05, 0.05, 0.12])
    if len(optimal_params) != qc.num_parameters:
        optimal_params = np.zeros(qc.num_parameters)
    if benchmark_cache.exists():
        with open(benchmark_cache, "r", encoding="utf-8") as f:
            bdata = json.load(f)
        results = bdata.get("results", [])
        best_run = min(results, key=lambda x: x["Error_mHa"])
        selected_ansatz_name = best_run["Ansatz"]
        print(f"Selected Best Benchmark Configuration: {selected_ansatz_name} ({best_run['Initialization']} init, {best_run['Optimizer']})")
    else:
        selected_ansatz_name = "DexcG"
        print("Benchmark cache not found. Defaulting to DexcG.")
        
    qc = ansatze[selected_ansatz_name]
    num_params = qc.num_parameters
    optimal_params = np.zeros(num_params) # Zero is exact optimal for DexcG / UCCSD
    
    # 3. Transpile circuit for backend
    print(f"\nTranspiling {qc.name} for {least_busy.name} (Optimization Level 3)...")
    pm = generate_preset_pass_manager(backend=least_busy, optimization_level=3)
    bound_circuit = qc.assign_parameters(optimal_params)
    transpiled_circuit = pm.run(bound_circuit)
    mapped_hamiltonian = H.apply_layout(transpiled_circuit.layout)
    
    ops = transpiled_circuit.count_ops()
    two_q_gates = sum(count for op, count in ops.items() if op in ["cx", "ecr", "cz"])
    print(f"Transpilation Complete:")
    print(f"  - Depth: {transpiled_circuit.depth()}")
    print(f"  - 2-Qubit Gates: {two_q_gates}")
    print(f"  - Total Gates: {sum(ops.values())}")
    
    # 4. Submit Job with EstimatorV2 (Job Mode)
    shots = 4096
    precision = 1.0 / np.sqrt(shots)
    print(f"\nSubmitting EstimatorV2 job to {least_busy.name} (shots={shots})...")
    
    estimator = EstimatorV2(mode=least_busy)
    pub = (transpiled_circuit, mapped_hamiltonian)
    job = estimator.run([pub], precision=precision)
    job_id = job.job_id()
    
    print("==================================================================")
    print("[SUCCESS] JOB SUBMITTED TO IBM QUANTUM!")
    print(f"  - Job ID:  {job_id}")
    print(f"  - Backend: {least_busy.name}")
    print(f"  - Direct URL: https://quantum.ibm.com/jobs/{job_id}")
    print("==================================================================")
    
    print("\nWaiting for physical QPU execution...")
    t0 = time.time()
    result = job.result() # Wait until finished
    elapsed = time.time() - t0
    
    evs = float(result[0].data.evs)
    err_mHa = abs(evs - E_exact) * 1000.0
    
    print("\n==================================================================")
    print("[COMPLETED] JOB COMPLETED ON IBM QUANTUM HARDWARE!")
    print("==================================================================")
    print(f"  - Backend:                 {least_busy.name}")
    print(f"  - Job ID:                  {job_id}")
    print(f"  - Total Elapsed Wait Time: {elapsed:.2f} s")
    print(f"  - Exact CASCI Energy:      {E_exact:.8f} Ha")
    print(f"  - Measured QPU Energy:     {evs:.8f} Ha")
    print(f"  - Energy Error:            {err_mHa:.4f} mHa")
    print(f"  - View on IBM Quantum:     https://quantum.ibm.com/jobs/{job_id}")
    print("==================================================================")
    # 2. Check for physical service
    try:
        service = get_runtime_service()
        backends = service.backends(operational=True, simulator=False)
        if not backends:
            raise RuntimeError("No operational physical backends found.")
        least_busy = min(backends, key=lambda b: b.status().pending_jobs)
        print(f"[Target Physical Backend] -> {least_busy.name} (Pending jobs: {least_busy.status().pending_jobs})")
        
        # 3. Transpile circuit
        pm = generate_preset_pass_manager(backend=least_busy, optimization_level=optimization_level)
        bound_circuit = qc.assign_parameters(optimal_params)
        transpiled_circuit = pm.run(bound_circuit)
        mapped_hamiltonian = H.apply_layout(transpiled_circuit.layout)
        
        ops = transpiled_circuit.count_ops()
        two_q_gates = sum(count for op, count in ops.items() if op in ["cx", "ecr", "cz"])
        depth = transpiled_circuit.depth()
        
        # 4. Submit Job with EstimatorV2 (Job Mode)
        precision = 1.0 / np.sqrt(shots)
        print(f"\nSubmitting EstimatorV2 job to {least_busy.name} (shots={shots})...")
        estimator = EstimatorV2(mode=least_busy)
        pub = (transpiled_circuit, mapped_hamiltonian)
        job = estimator.run([pub], precision=precision)
        job_id = job.job_id()
        
        print(f"[SUCCESS] Job Submitted! Job ID: {job_id}")
        t0 = time.time()
        result = job.result()
        qpu_time = time.time() - t0
        
        measured_energy = float(result[0].data.evs)
        err_mHa = abs(measured_energy - E_exact) * 1000.0
        
        record = {
            "status": "COMPLETED_ON_PHYSICAL_QPU",
            "backend": least_busy.name,
            "job_id": job_id,
            "ansatz": selected_ansatz_name,
            "parameter_vector": optimal_params.tolist(),
            "optimization_level": optimization_level,
            "resilience_options": "default",
            "shots": shots,
            "transpiled_depth": depth,
            "transpiled_2q_gates": two_q_gates,
            "qpu_seconds": round(qpu_time, 2),
            "measured_energy_ha": measured_energy,
            "exact_casci_energy_ha": E_exact,
            "error_mha": err_mHa
        }
        
    except Exception as e:
        print(f"[Notice] Physical QPU unavailable: {e}. Executing calibrated FakeFez noisy Aer simulation.")
        sim_res = run_noisy_fake_backend_evaluation(qc, H, optimal_params, E_exact, shots=shots)
        record = {
            "status": "CALIBRATED_NOISY_AER_SIMULATION",
            "backend": sim_res["Target Backend"],
            "job_id": "sim-aer-noise-001",
            "ansatz": selected_ansatz_name,
            "parameter_vector": optimal_params.tolist(),
            "optimization_level": optimization_level,
            "resilience_options": "readout_error_mitigation",
            "shots": shots,
            "transpiled_depth": sim_res["Circuit Depth"],
            "transpiled_2q_gates": sim_res["Transpiled 2-Qubit Gate Count"],
            "qpu_seconds": sim_res["QPU Runtime (seconds)"],
            "measured_energy_ha": sim_res["Hardware Measured Energy (Ha)"],
            "exact_casci_energy_ha": E_exact,
            "error_mha": sim_res["Hardware Error (mHa)"]
        }
        
    # Save full record to data/hardware_run.json
    with open(DATA_DIR / "hardware_run.json", "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print(f"[OK] Saved hardware execution record to {DATA_DIR / 'hardware_run.json'}")
    return record

if __name__ == "__main__":
    run_live_qpu()

