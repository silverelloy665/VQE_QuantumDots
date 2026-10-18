"""
IBM Quantum Hardware evaluation module for BN Quantum Dot VQE benchmark.
Executes 1 single-point expectation evaluation on ibm_marrakesh (Heron r2, 156q)
in job mode (no sessions), utilising the full remaining QPU budget.

QPU target: ibm_marrakesh (open-instance, Heron r2, 156q)
"""
import time
import numpy as np
import pandas as pd
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2

from src.config import get_runtime_service

_DEFAULT_BACKEND = "ibm_marrakesh"
_JOB_TIMEOUT_S = 220


def are_paulis_qwc(p1: str, p2: str) -> bool:
    """
    Check if two Pauli strings are qubit-wise commuting (QWC).
    Two Pauli operators qubit-wise commute iff on every qubit index they commute,
    i.e., at least one is 'I' or both are identical.
    """
    return all(c1 == "I" or c2 == "I" or c1 == c2 for c1, c2 in zip(p1, p2))


def group_paulis_qwc(hamiltonian: SparsePauliOp) -> list[SparsePauliOp]:
    """
    Groups the Pauli terms of the Hamiltonian into qubit-wise commuting (QWC) groups
    using greedy graph coloring.
    
    Parameters
    ----------
    hamiltonian : SparsePauliOp
        The active-space qubit Hamiltonian.
        
    Returns
    -------
    list[SparsePauliOp]
        List of SparsePauliOp objects, each representing a commuting group of terms.
    """
    labels = hamiltonian.paulis.to_labels()
    coeffs = hamiltonian.coeffs
    
    groups: list[list[str]] = []
    group_coeffs: list[list[complex]] = []
    
    for label, coeff in zip(labels, coeffs):
        placed = False
        for g_labels, g_coeffs in zip(groups, group_coeffs):
            if all(are_paulis_qwc(label, existing) for existing in g_labels):
                g_labels.append(label)
                g_coeffs.append(coeff)
                placed = True
                break
        if not placed:
            groups.append([label])
            group_coeffs.append([coeff])
            
    group_ops = [SparsePauliOp(g, c) for g, c in zip(groups, group_coeffs)]
    return group_ops


def run_hardware_evaluation(
    circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    optimal_params: np.ndarray,
    exact_energy: float,
    max_2q_gates: int = 300,
    shots: int = 8192,
    backend_name: str | None = _DEFAULT_BACKEND,
) -> dict:
    """
    Runs a single-point energy evaluation on physical IBM QPU.
    Strictly reports real hardware results, returning 'NOT RUN' on any failure.
    No fabricated or estimated fallback numbers.
    """
    print("------------------------------------------------------------------")
    print("Initiating IBM Quantum Hardware Execution Step...")
    print(f"  Target QPU  : {backend_name or 'least-busy'}")
    print(f"  Shots       : {shots}")
    print(f"  Job timeout : {_JOB_TIMEOUT_S} s")
    print("------------------------------------------------------------------")

    try:
        service = get_runtime_service()
    except Exception as e:
        print(f"[Warning] Could not initialize QiskitRuntimeService: {e}.")
        return {"Status": "NOT RUN", "reason": f"Could not initialize QiskitRuntimeService: {e}"}

    # 1. Resolve backend
    backend = None
    if backend_name:
        try:
            backend = service.backend(backend_name)
            if not backend.status().operational:
                print(f"[Warning] {backend_name} is not operational right now. Falling back to least-busy.")
                backend = None
        except Exception as exc:
            print(f"[Warning] Could not fetch {backend_name}: {exc}. Falling back to least-busy.")

    if backend is None:
        try:
            backends = service.backends(operational=True, simulator=False)
            backend = min(backends, key=lambda b: b.status().pending_jobs) if backends else None
        except Exception as e:
            return {"Status": "NOT RUN", "reason": f"Failed to list physical backends: {e}"}

    if backend is None:
        print("[Warning] No operational physical QPUs available.")
        return {"Status": "NOT RUN", "reason": "No operational physical QPUs available"}

    pending = backend.status().pending_jobs
    print(f"Selected QPU: {backend.name} (Pending jobs: {pending})")

    # 2. Transpile and map Hamiltonian onto backend layout
    try:
        pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
        bound_circuit = circuit.assign_parameters(optimal_params)
        transpiled_circuit = pm.run(bound_circuit)
        mapped_hamiltonian = hamiltonian.apply_layout(transpiled_circuit.layout)

        ops = transpiled_circuit.count_ops()
        t_2q_gates = sum(count for op, count in ops.items() if op in ["cx", "ecr", "cz"])
        t_depth = transpiled_circuit.depth()
        print(f"Transpiled for {backend.name}: Depth = {t_depth}, 2-Qubit Gates = {t_2q_gates}")
    except Exception as e:
        return {"Status": "NOT RUN", "reason": f"Transpilation failed: {e}"}

    if t_2q_gates > max_2q_gates:
        print(f"[Constraint] 2-Qubit gate count ({t_2q_gates}) exceeds limit ({max_2q_gates}).")

    # 3. Submit EstimatorV2 job
    print(f"Submitting single-point EstimatorV2 job to {backend.name} (shots={shots})...")
    t0 = time.perf_counter()
    try:
        estimator = EstimatorV2(mode=backend)
        pub = (transpiled_circuit, mapped_hamiltonian)
        job = estimator.run([pub], precision=1.0 / np.sqrt(shots))
        job_id = job.job_id()
        print(f"  Job ID: {job_id} — waiting up to {_JOB_TIMEOUT_S} s ...")
        
        result = job.result(timeout=_JOB_TIMEOUT_S)
        measured_energy = float(result[0].data.evs)
        qpu_time = time.perf_counter() - t0
        error_mHa = abs(measured_energy - exact_energy) * 1000.0
        print(f"Job completed on QPU! Measured Energy: {measured_energy:.8f} Ha | Error: {error_mHa:.4f} mHa")

        return {
            "Status": "Completed on Physical QPU",
            "Target Backend": backend.name,
            "Job ID": job_id,
            "Ansatz Selected": circuit.name or "VQE Ansatz",
            "Parameters Optimized": len(optimal_params),
            "Transpiled 2-Qubit Gate Count": t_2q_gates,
            "Circuit Depth": t_depth,
            "Exact Active Ground Energy (Ha)": exact_energy,
            "Hardware Measured Energy (Ha)": measured_energy,
            "Hardware Error (mHa)": error_mHa,
            "Shots": shots,
            "QPU Runtime (seconds)": round(qpu_time, 2),
        }
    except Exception as timeout_or_err:
        print(f"[Error] QPU execution error: {timeout_or_err}")
        return {
            "Status": "NOT RUN",
            "Target Backend": backend.name,
            "reason": str(timeout_or_err),
            "Transpiled 2-Qubit Gate Count": t_2q_gates,
            "Circuit Depth": t_depth
        }


def get_simulated_hardware_metrics(*args, **kwargs) -> dict:
    """Mock/fallback metrics when no physical QPU is run."""
    return {"Status": "NOT RUN", "reason": "No physical QPU run requested"}


if __name__ == "__main__":
    from src.molecule import load_or_build_bn_dot_hamiltonian
    H, _, _ = load_or_build_bn_dot_hamiltonian()
    groups = group_paulis_qwc(H)
    print(f"QWC grouping: {len(H.paulis)} Pauli terms grouped into {len(groups)} commuting circuits.")
