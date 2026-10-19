"""
Unit tests for convergence and iteration flags:
Verifies that maxiter=1 correctly sets hit_maxiter=True and converged=False.
"""
import pytest
import numpy as np
from src.molecule import load_hard_hamiltonian
from src.ansatze import get_ansatz_dict
from src.optimizers import run_vqe_single


@pytest.fixture(scope="module")
def hard_problem_fixture():
    H, E_exact, meta = load_hard_hamiltonian()
    ansatze = get_ansatz_dict(num_spatial_orbitals=4, num_particles=(2, 2), include_k_variants=True)
    return H, E_exact, ansatze


@pytest.mark.parametrize("optimizer", ["GD", "ADAM", "COBYLA", "L-BFGS-B", "SPSA"])
def test_maxiter_1_reports_hit_maxiter_and_unconverged(hard_problem_fixture, optimizer):
    H, E_exact, ansatze = hard_problem_fixture
    qc = ansatze["1-UpCCGSD"].decompose().decompose()

    res = run_vqe_single(
        circuit=qc,
        hamiltonian=H,
        ansatz_name="1-UpCCGSD",
        init_name="random",
        optimizer_name=optimizer,
        maxiter=1,
        seed=42
    )

    assert res["hit_maxiter"] is True, f"{optimizer} with maxiter=1 should set hit_maxiter=True"
    assert res["converged"] is False, f"{optimizer} with maxiter=1 should set converged=False"
    assert "final_params" in res and len(res["final_params"]) == qc.num_parameters
    assert "best_params" in res and len(res["best_params"]) == qc.num_parameters
    assert res["iterations_used"] >= 1


def test_rotosolve_reports_converged_and_not_hit_maxiter(hard_problem_fixture):
    from src.molecule import load_or_build_bn_dot_hamiltonian
    H_2e2o, _, _ = load_or_build_bn_dot_hamiltonian("6-31g(d,p)")
    ansatze = get_ansatz_dict()
    qc = ansatze["DexcG"]

    res = run_vqe_single(
        circuit=qc,
        hamiltonian=H_2e2o,
        ansatz_name="DexcG",
        init_name="zero",
        optimizer_name="ROTOSOLVE"
    )

    assert res["converged"] is True
    assert res["hit_maxiter"] is False
    assert res["success"] is True
    assert res["iterations_used"] == 1

