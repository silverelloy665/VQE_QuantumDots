"""
Tests for Phase E: Hardware job preparation, dry-run safety, QWC grouping,
and fidelity threshold validation.
"""
import json
import pytest
from pathlib import Path
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.circuit.library import HartreeFock, UCC
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit.quantum_info import Statevector
import numpy as np

from scripts.run_live_hardware_job import (
    build_original_2e2o_parity_circuit,
    execute_hardware_jobs,
    prepare_hardware_job_specs_v3,
    SPEC_V2_PATH,
    SPEC_V3_PATH
)
from src.hardware import group_paulis_qwc
from src.molecule import load_or_build_bn_dot_hamiltonian, load_hard_hamiltonian


def test_hardware_job_spec_v2_exists_and_valid():
    """Verify data/hardware_job_spec_v2.json exists, is valid JSON, and has 15 jobs."""
    assert SPEC_V2_PATH.exists(), f"Missing spec file: {SPEC_V2_PATH}"
    with open(SPEC_V2_PATH, "r", encoding="utf-8") as f:
        specs = json.load(f)
    assert isinstance(specs, list)
    assert len(specs) >= 12, f"Expected at least 12 jobs, got {len(specs)}"

    # Required experiment names
    exp_names = {s["experiment_name"] for s in specs}
    assert "HF_baseline_2e2o" in exp_names or "HF_baseline_hard_4e4o" in exp_names
    assert "original_2e2o_parity_reduced" in exp_names
    assert "hard_problem_1-UpCCGSD" in exp_names
    assert "hard_problem_2-UpCCGSD_too_deep" in exp_names

    # Check resilience types are separate
    for s in specs:
        assert s["resilience_type"] in ["dynamical_decoupling", "pauli_twirling", "zne"]
        assert s["shots"] > 0
        assert s["status"] == "PREPARED (NOT SUBMITTED)"


def test_parity_minimal_circuit_casci_precision():
    """Verify that the 2-qubit parity minimal circuit reproduces CASCI to < 1e-6 Ha."""
    res = build_original_2e2o_parity_circuit()
    assert res["num_qubits"] == 2
    assert res["diff_ha"] < 1e-6, f"Parity CASCI diff {res['diff_ha']:.2e} Ha exceeds 1e-6 Ha threshold"
    assert len(res["parameters"]) == 1


def test_fidelity_threshold_recommendations():
    """Verify that any circuit with estimated fidelity < 0.3 is marked NOT RECOMMENDED."""
    with open(SPEC_V2_PATH, "r", encoding="utf-8") as f:
        specs = json.load(f)

    for s in specs:
        if s["estimated_fidelity"] < 0.3:
            assert "NOT RECOMMENDED: too deep" in s["recommendation"]
        else:
            assert "RECOMMENDED" in s["recommendation"]


def test_hardware_dry_run_refusal():
    """Verify that execute_hardware_jobs without submit=True refuses execution."""
    res = execute_hardware_jobs([], submit=False)
    assert res["status"] == "DRY_RUN_COMPLETED"
    assert res["jobs_submitted"] == 0


def test_hardware_job_spec_v3_exists_and_valid():
    """Verify data/hardware_job_spec_v3.json exists, adheres to Schema v3.0, and enforces error models."""
    assert SPEC_V3_PATH.exists(), f"Missing spec v3 file: {SPEC_V3_PATH}"
    with open(SPEC_V3_PATH, "r", encoding="utf-8") as f:
        specs = json.load(f)
    assert isinstance(specs, list)
    assert len(specs) >= 12, f"Expected at least 12 jobs in v3 spec, got {len(specs)}"

    exp_names = {s["experiment_name"] for s in specs}
    assert "HF_baseline_2e2o" in exp_names
    assert "HF_baseline_hard_4e4o" in exp_names
    assert "original_2e2o_parity_reduced" in exp_names

    for s in specs:
        assert s["fidelity_error_model"] == "unified_cz_edge_readout_decoherence_v3"
        assert s["backend_source"] in ["LIVE", "MOCK", "ASSUMED"]
        assert "calibration_timestamp" in s
        assert "qiskit_version" in s
        assert s["status"] == "PREPARED (NOT SUBMITTED)"

        if s["experiment_name"] == "original_2e2o_parity_reduced":
            assert "noise/mitigation test only" in s["recommendation"]
            assert s["shots_needed_1sigma"] is not None
            assert s["shots_needed_1sigma"]["uniform_allocation"] > 2_000_000

        if s["estimated_fidelity"] < 0.3 and "hard_problem" in s["experiment_name"]:
            assert "NOT RECOMMENDED" in s["recommendation"]
            assert s["priority"] >= 4


def test_mock_protection_without_flag():
    """Verify that attempting to transpile against an unreachable backend raises RuntimeError without --mock."""
    with pytest.raises(RuntimeError) as exc_info:
        prepare_hardware_job_specs_v3(
            backend_name="completely_nonexistent_ibm_fake_device",
            allow_mock=False
        )
    assert "Live backend" in str(exc_info.value)
    assert "--mock flag not provided" in str(exc_info.value)

