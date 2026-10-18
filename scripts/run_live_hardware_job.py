"""
Phase 6 IBM Quantum Hardware Cross-Check Script.

Target:
  - B8N8H10 hexagonal boron-nitride quantum dot (2e, 2o active space, 4 qubits).
  - Evaluates an energy landscape (5-6 points) around theta=0 to test if physical QPU
    reproduces the shape of the potential energy curve.

Strict Safety & Guardrails:
  1. Targets the cheapest circuit: compares PCU2 and DexcG at opt level 3; picks shallower.
  2. Strict Pre-Submission Approval Gate:
     - Prints backend name, transpiled depth, and 2-qubit gate count.
     - Hard exits if 2-qubit gate count > 60.
     - Pauses with input() confirmation; aborts if not 'y'.
  3. Uses EstimatorV2 in Job Mode (no sessions, Open Plan compliant).
  4. resilience_level=0 to minimize QPU time overhead.
  5. Logs comparison of ideal statevector vs noisy QPU energies to data/hardware_crosscheck.json.
"""
import sys
import os
import time
import json
import argparse
from datetime import datetime, timezone
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from src.config import get_runtime_service
from src.molecule import load_or_build_bn_dot_hamiltonian
from src.ansatze import get_ansatz_dict

DATA_DIR = REPO_ROOT / "data"
CROSSCHECK_CACHE = DATA_DIR / "hardware_crosscheck.json"
MAX_2Q_GATE_LIMIT = 60  # Strict gate count safety threshold


def count_2q_gates(circuit: QuantumCircuit) -> int:
    """Counts two-qubit entangling gates (cx, cz, ecr) in a circuit."""
    ops = circuit.count_ops()
    return sum(count for op, count in ops.items() if op in ["cx", "cz", "ecr"])


def build_landscape_points(ansatz_name: str, num_params: int) -> tuple[np.ndarray, list[str]]:
    """
    Constructs an array of 5 to 6 parameter points representing the energy landscape.
    Includes theta = 0 (Hartree-Fock), near-optimal point, and displaced points.
    """
    if ansatz_name == "DexcG" and num_params == 1:
        # 1D Double Excitation parameter sweep around 0
        # Optimal parameter from benchmark is theta* ~ +0.0045 rad
        thetas = np.array([
            [-0.10],
            [-0.05],
            [0.00],       # Exact Hartree-Fock reference
            [0.0045],     # Near-optimal variational minimum
            [+0.05],
            [+0.10]
        ], dtype=float)
        labels = [
            "theta = -0.100 rad",
            "theta = -0.050 rad",
            "theta =  0.000 rad (HF Ref)",
            "theta = +0.0045 rad (VQE Min)",
            "theta = +0.050 rad",
            "theta = +0.100 rad"
        ]
        return thetas, labels

    elif ansatz_name == "PCU2":
        # PCU2 has 14 parameters. We sweep along the primary entangling excitation
        # (parameter 0) while keeping other parameters at 0.
        points = []
        labels = []
        disp_vals = [-0.10, -0.05, 0.00, +0.05, +0.10]
        for val in disp_vals:
            vec = np.zeros(num_params, dtype=float)
            vec[0] = val
            points.append(vec)
            if val == 0.0:
                labels.append(f"theta_0 = {val:+.3f} (HF Ref, zeros)")
            else:
                labels.append(f"theta_0 = {val:+.3f} (displaced)")
        return np.array(points, dtype=float), labels

    else:
        # Generic uniform displacement sweep
        points = []
        labels = []
        for val in [-0.10, -0.05, 0.00, +0.05, +0.10]:
            points.append(np.full(num_params, val, dtype=float))
            labels.append(f"all_params = {val:+.3f}")
        return np.array(points, dtype=float), labels


def compute_ideal_energies(
    circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    points: np.ndarray
) -> np.ndarray:
    """Computes exact noiseless Statevector energies for the parameter points."""
    estimator = StatevectorEstimator()
    pub = (circuit, hamiltonian, points)
    job = estimator.run([pub])
    return np.array(job.result()[0].data.evs, dtype=float)


def run_hardware_crosscheck(
    backend_override: str | None = None,
    ansatz_choice: str = "auto",
    shots: int = 4096,
    dry_run: bool = False
):
    print("=" * 78)
    print("PHASE 6: IBM QUANTUM HARDWARE CROSS-CHECK (OPEN PLAN SAFE HARNESS)")
    print("=" * 78)

    # 1. Load Hamiltonian for BN Quantum Dot (2e, 2o)
    print("\n[Step 1/6] Loading (2e, 2o) active space Hamiltonian...")
    H, E_exact, meta = load_or_build_bn_dot_hamiltonian("6-31g(d,p)", 2, 2)
    ref_det_energy = meta.get("reference_determinant_energy", meta["hf_energy"])
    print(f"  System: {meta['label']} | Basis: 6-31G(d,p)")
    print(f"  Mapped Qubits: {H.num_qubits} | Pauli Terms: {len(H)}")
    print(f"  Reference Determinant Energy: {ref_det_energy:.8f} Ha")
    print(f"  Exact CASCI Energy:          {E_exact:.8f} Ha")
    print(f"  Active Space E_corr:         {meta['correlation_energy_mHa']:.5f} mHa")

    # 2. Select Backend
    print("\n[Step 2/6] Connecting to IBM Quantum Service & selecting backend...")
    if dry_run:
        print("  [DRY-RUN] Simulating backend selection using local FakeFez (156-qubit Heron)...")
        from src.hardware import get_fake_backend
        backend = get_fake_backend("ibm_fez")
        backend_name = "ibm_fez (Simulated Heron r2)"
    else:
        service = get_runtime_service()
        if backend_override:
            backend = service.backend(backend_override)
            print(f"  Target Backend (manual override): {backend.name}")
        else:
            print("  Querying least busy physical backend (operational=True, min_num_qubits=4)...")
            backend = service.least_busy(operational=True, simulator=False, min_num_qubits=4)
            print(f"  Least Busy Backend Found: {backend.name}")
        backend_name = backend.name

    print(f"  Backend: {backend_name} ({getattr(backend, 'num_qubits', 'unknown')} qubits)")
    if hasattr(backend, "status"):
        status = backend.status()
        print(f"  Pending Jobs: {getattr(status, 'pending_jobs', 'N/A')} | Operational: {getattr(status, 'operational', True)}")

    # 3. Target the Cheapest Circuit (Compare PCU2 vs DexcG)
    print("\n[Step 3/6] Transpiling candidate ansatze (PCU2 vs DexcG) at optimization level 3...")
    ansatze = get_ansatz_dict(num_spatial_orbitals=2, num_particles=(1, 1))
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)

    candidates = {}
    for name in ["PCU2", "DexcG"]:
        raw_qc = ansatze[name].decompose()
        isa_qc = pm.run(raw_qc)
        depth = isa_qc.depth()
        two_q = count_2q_gates(isa_qc)
        candidates[name] = {
            "raw_circuit": raw_qc,
            "isa_circuit": isa_qc,
            "depth": depth,
            "2q_gates": two_q,
            "num_params": raw_qc.num_parameters
        }
        print(f"  Candidate {name:6s} -> Transpiled Depth: {depth:3d} | 2-Qubit Gates: {two_q:2d} | Parameters: {raw_qc.num_parameters:2d}")

    # Pick whichever transpiles shallower (or use user override)
    if ansatz_choice.lower() in ["dexcg", "pcu2"]:
        chosen_name = "DexcG" if ansatz_choice.lower() == "dexcg" else "PCU2"
    else:
        chosen_name = min(candidates.keys(), key=lambda k: (candidates[k]["depth"], candidates[k]["2q_gates"]))

    chosen_info = candidates[chosen_name]
    isa_circuit = chosen_info["isa_circuit"]
    chosen_depth = chosen_info["depth"]
    chosen_2q = chosen_info["2q_gates"]

    print(f"\n  -> SELECTED ANSATZ: {chosen_name} (depth {chosen_depth}, {chosen_2q} 2Q gates)")

    # 4. Build Energy Landscape Points & Compute Ideal Statevector Energies
    print("\n[Step 4/6] Constructing parameter sweep landscape & computing ideal Statevector energies...")
    points, labels = build_landscape_points(chosen_name, chosen_info["num_params"])
    ideal_energies = compute_ideal_energies(chosen_info["raw_circuit"], H, points)

    print(f"  Generated {len(points)} landscape evaluation points:")
    for idx, (lbl, val, e_ideal) in enumerate(zip(labels, points, ideal_energies), 1):
        print(f"    Point {idx}: {lbl:32s} -> Ideal Energy: {e_ideal:.8f} Ha")

    # Map observable to backend ISA layout
    isa_observable = H.apply_layout(isa_circuit.layout)

    # 5. STRICT PRE-SUBMISSION APPROVAL GATE
    print("\n" + "=" * 78)
    print("STRICT PRE-SUBMISSION APPROVAL GATE")
    print("=" * 78)
    print(f"  Target Backend:             {backend_name}")
    print(f"  Ansatz:                     {chosen_name}")
    print(f"  Parameters per point:       {chosen_info['num_params']}")
    print(f"  Evaluation Points:          {len(points)}")
    print(f"  Transpiled Circuit Depth:   {chosen_depth}")
    print(f"  Transpiled 2-Qubit Gates:   {chosen_2q} (limit: {MAX_2Q_GATE_LIMIT})")
    print(f"  Shots per point:            {shots}")
    print(f"  Total Quantum Shots:        {shots * len(points):,}")
    print(f"  Execution Mode:             EstimatorV2 Job Mode (resilience_level=0)")
    print(f"  Estimated QPU Time:         ~15 to 35 seconds (well within 600s quota)")
    print("=" * 78)

    # Gate Count Hard Limit Check
    if chosen_2q > MAX_2Q_GATE_LIMIT:
        print(f"\n[FATAL ERROR] Transpiled 2-qubit gate count ({chosen_2q}) EXCEEDS safety limit of {MAX_2Q_GATE_LIMIT}!")
        print("Aborting submission immediately to protect QPU quota.")
        sys.exit(1)

    if dry_run:
        print("\n[DRY-RUN MODE] Pre-submission approval check passed.")
        print("Dry run completed successfully. Skipping live hardware submission.")
        return

    # Interactive Prompt Gate
    response = input("\nProceed with hardware submission? (y/n): ")
    if response.strip().lower() != "y":
        print("\n[SUBMISSION CANCELLED] User did not confirm submission ('y'). Exiting safely.")
        sys.exit(0)

    # 6. Execute via EstimatorV2 in Job Mode
    print("\n[Step 5/6] Submitting job to physical IBM QPU via EstimatorV2...")
    from qiskit_ibm_runtime import EstimatorV2

    estimator = EstimatorV2(mode=backend)
    estimator.options.resilience_level = 0
    estimator.options.default_shots = shots

    pub = (isa_circuit, isa_observable, points)
    t_submit0 = time.time()
    job = estimator.run([pub])
    job_id = job.job_id()

    print(f"\n>>> JOB SUBMITTED SUCCESSFULLY! <<<")
    print(f"  Job ID:        {job_id}")
    print(f"  Target QPU:    {backend.name}")
    print(f"  Live Tracking: https://quantum.ibm.com/jobs/{job_id}")
    print("\nWaiting for job to complete (polling status)...")

    # Monitor until completion
    job_result = job.result()
    t_complete = time.time()
    wall_duration = t_complete - t_submit0

    print(f"\n[DONE] Job completed in {wall_duration:.1f} s wall time!")

    # 7. Extract Results, Metrics & Query Quantum Seconds
    print("\n[Step 6/6] Extracting QPU metrics & formatting energy landscape...")
    pub_result = job_result[0]
    noisy_energies = np.array(pub_result.data.evs, dtype=float)

    # Extract standard deviations if available
    if hasattr(pub_result.data, "stds") and pub_result.data.stds is not None:
        stds = np.array(pub_result.data.stds, dtype=float)
    else:
        stds = np.full_like(noisy_energies, np.nan)

    # Query billed quantum_seconds from metrics
    try:
        metrics = job.metrics()
        qpu_seconds = metrics.get("usage", {}).get("quantum_seconds", metrics.get("quantum_seconds", 0.0))
    except Exception:
        qpu_seconds = 0.0

    print(f"  Billed Quantum Seconds: {qpu_seconds:.2f} s")

    # Build comparison records
    crosscheck_records = []
    print("\n" + "-" * 90)
    print(f"{'Point':<8s} | {'Label':<32s} | {'Ideal Energy (Ha)':<18s} | {'QPU Energy (Ha)':<18s} | {'Err (mHa)':<10s}")
    print("-" * 90)

    for idx, (lbl, e_ideal, e_qpu, std_val) in enumerate(zip(labels, ideal_energies, noisy_energies, stds), 1):
        err_mHa = abs(e_qpu - e_ideal) * 1000.0
        print(f"Point {idx:<2d} | {lbl:<32s} | {e_ideal:18.8f} | {e_qpu:18.8f} | {err_mHa:10.4f}")
        crosscheck_records.append({
            "point_index": idx,
            "label": lbl,
            "ideal_energy_ha": float(e_ideal),
            "noisy_qpu_energy_ha": float(e_qpu),
            "error_mha": float(err_mHa),
            "std_dev_ha": float(std_val) if not np.isnan(std_val) else None
        })
    print("-" * 90)

    # Save to data/hardware_crosscheck.json
    crosscheck_data = {
        "status": "COMPLETED_ON_PHYSICAL_QPU",
        "backend": backend.name,
        "job_id": job_id,
        "job_dashboard": f"https://quantum.ibm.com/jobs/{job_id}",
        "ansatz": chosen_name,
        "transpiled_depth": chosen_depth,
        "transpiled_2q_gates": chosen_2q,
        "shots_per_point": shots,
        "total_points": len(points),
        "total_shots": shots * len(points),
        "resilience_level": 0,
        "quantum_seconds_billed": float(qpu_seconds),
        "wall_duration_seconds": float(wall_duration),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "landscape_comparison": crosscheck_records
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CROSSCHECK_CACHE, "w", encoding="utf-8") as f:
        json.dump(crosscheck_data, f, indent=2)

    print(f"\nCross-check results saved to: {CROSSCHECK_CACHE}")
    print("Phase 6 IBM QPU Cross-Check successfully completed!")


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 6 IBM Quantum Hardware Cross-Check")
    parser.add_argument("--backend", type=str, default=None, help="Target backend name (default: least busy)")
    parser.add_argument("--ansatz", type=str, default="auto", choices=["auto", "PCU2", "DexcG"], help="Ansatz choice: auto (cheapest), PCU2, or DexcG")
    parser.add_argument("--shots", type=int, default=4096, help="Shots per evaluation point (default: 4096)")
    parser.add_argument("--dry-run", action="store_true", help="Simulate transpilation & checks without live QPU submission")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_hardware_crosscheck(
        backend_override=args.backend,
        ansatz_choice=args.ansatz,
        shots=args.shots,
        dry_run=args.dry_run
    )
