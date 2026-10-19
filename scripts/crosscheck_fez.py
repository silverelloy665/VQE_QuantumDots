"""
Crosscheck physical QPU evaluation script on ibm_fez (Heron r2, 156q).
Submits and evaluates the ground-state active-space energy of B8N8H10
under identical conditions to ibm_marrakesh job dbahr3hgv2cc73e6stt0 (8192 shots, opt level 3).

Saves result to data/hardware_crosscheck_fez.json and compares performance across Heron QPUs.
"""
import os
import sys
import json
import time
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2

from src.config import get_runtime_service
from src.molecule import load_or_build_bn_dot_hamiltonian
from src.ansatze import get_ansatz_dict

DATA_DIR = REPO_ROOT / "data"
OUTPUT_PATH = DATA_DIR / "hardware_crosscheck_fez.json"

TARGET_BACKEND = "ibm_fez"
SHOTS = 8192
TIMEOUT_S = 300


def main():
    print("=" * 75)
    print("CROSSCHECK ON IBM FEZ (HERON R2, 156Q)")
    print(f"Target Backend: {TARGET_BACKEND} | Shots: {SHOTS}")
    print("=" * 75)

    service = get_runtime_service()
    backend = service.backend(TARGET_BACKEND)
    st = backend.status()
    print(f"Backend Status: operational={st.operational}, pending_jobs={st.pending_jobs}, status_msg={st.status_msg}")

    if not st.operational:
        raise RuntimeError(f"Backend {TARGET_BACKEND} is currently not operational ({st.status_msg}).")

    # Load B8N8H10 2e2o Hamiltonian and reference CASCI
    H, E_casci, meta = load_or_build_bn_dot_hamiltonian("6-31g(d,p)")
    ansatze = get_ansatz_dict()
    qc = ansatze["UCCSD"]
    params = np.zeros(qc.num_parameters)

    print(f"\nAnsatz: UCCSD (4 qubits, {qc.num_parameters} parameters, zero-init)")
    print(f"Exact CASCI Reference: {E_casci:.8f} Ha")

    # Transpile
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3, seed_transpiler=42)
    bound_qc = qc.assign_parameters(params)
    transpiled_qc = pm.run(bound_qc)
    mapped_H = H.apply_layout(transpiled_qc.layout)

    ops = transpiled_qc.count_ops()
    cz_count = ops.get("cz", 0) + ops.get("ecr", 0) + ops.get("cx", 0)
    depth = transpiled_qc.depth()
    print(f"Transpiled for {TARGET_BACKEND}: Depth = {depth}, 2-Qubit (CZ) Gates = {cz_count}")

    # Submit job
    print(f"\nSubmitting EstimatorV2 job to {TARGET_BACKEND} (mode=backend, shots={SHOTS})...")
    estimator = EstimatorV2(mode=backend)
    pub = (transpiled_qc, mapped_H)
    t0 = time.perf_counter()
    job = estimator.run([pub], precision=1.0 / np.sqrt(SHOTS))
    job_id = job.job_id()
    job_url = f"https://quantum.ibm.com/jobs/{job_id}"

    print(f"  Job Submitted Successfully!")
    print(f"  Job ID : {job_id}")
    print(f"  URL    : {job_url}")
    print(f"Waiting up to {TIMEOUT_S} s for job completion on physical QPU...")

    record = {
        "status": "QUEUED_ON_PHYSICAL_QPU",
        "backend": TARGET_BACKEND,
        "job_id": job_id,
        "job_url": job_url,
        "shots": SHOTS,
        "transpiled_cz_gates": cz_count,
        "transpiled_depth": depth,
        "exact_casci_ha": E_casci,
        "marrakesh_comparison_job": "dbahr3hgv2cc73e6stt0",
        "marrakesh_measured_ha": -639.65177323,
        "marrakesh_error_mha": 72.5100,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

    try:
        result = job.result(timeout=TIMEOUT_S)
        wall_time = time.perf_counter() - t0
        measured_e = float(result[0].data.evs)
        error_mha = abs(measured_e - E_casci) * 1000.0

        print("\n" + "=" * 75)
        print("PHYSICAL JOB COMPLETED ON IBM FEZ!")
        print(f"  Measured Energy : {measured_e:.8f} Ha")
        print(f"  Exact CASCI     : {E_casci:.8f} Ha")
        print(f"  Hardware Error  : {error_mha:.4f} mHa")
        print(f"  Elapsed Time    : {wall_time:.2f} s")
        print("=" * 75)

        record.update({
            "status": "COMPLETED_ON_PHYSICAL_QPU",
            "measured_energy_ha": measured_e,
            "error_mha": error_mha,
            "wall_time_seconds": round(wall_time, 2),
            "fez_vs_marrakesh_delta_mha": round(error_mha - 72.5100, 4)
        })

    except Exception as exc:
        print(f"\n[Job Update] Job {job_id} status: {job.status()} ({exc})")
        record.update({
            "status": str(job.status()),
            "reason": str(exc),
            "wall_time_seconds": round(time.perf_counter() - t0, 2)
        })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print(f"\nSaved crosscheck execution record to {OUTPUT_PATH}")
    return record


if __name__ == "__main__":
    main()

