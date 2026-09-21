"""
Submit and monitor a live IBM Quantum Hardware evaluation job on least-busy QPU.
"""
import sys
import time
import numpy as np
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2

from src.config import get_runtime_service
from src.molecule import load_or_build_bn_dot_hamiltonian
from src.ansatze import get_ansatz_dict

def run_live_qpu():
    print("==================================================================")
    print("SUBMITTING LIVE IBM QUANTUM HARDWARE JOB")
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
    H, E_exact, meta = load_or_build_bn_dot_hamiltonian("6-31g(d,p)")
    ansatze = get_ansatz_dict()
    qc = ansatze["UCCSD"]
    
    # Let's use the optimized parameters for UCCSD (or near-optimal small angle)
    optimal_params = np.array([0.05, 0.05, 0.12])
    if len(optimal_params) != qc.num_parameters:
        optimal_params = np.zeros(qc.num_parameters)
    
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

if __name__ == "__main__":
    run_live_qpu()

