"""
Verification Test: Particle Number (N=2) and Spin Projection (Sz=0) conservation on optimized states.
"""
import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp, Statevector
from src.molecule import load_or_build_bn_dot_hamiltonian
from src.ansatze import get_ansatz_dict
from src.optimizers import run_vqe_single

@pytest.fixture(scope="module")
def setup_system():
    H, E_exact, meta = load_or_build_bn_dot_hamiltonian("6-31g(d,p)")
    ansatze = get_ansatz_dict()
    
    # 4 qubits: q0=alpha0, q1=alpha1, q2=beta0, q3=beta1
    # N = sum (I - Zq)/2 = 2*I - 0.5*(Z0 + Z1 + Z2 + Z3)
    N_op = SparsePauliOp(["IIII", "IIIZ", "IIZI", "IZII", "ZIII"], [2.0, -0.5, -0.5, -0.5, -0.5])
    # Sz = 0.5 * (N_alpha - N_beta) = 0.25 * (-Z0 - Z1 + Z2 + Z3)
    Sz_op = SparsePauliOp(["IIIZ", "IIZI", "IZII", "ZIII"], [-0.25, -0.25, 0.25, 0.25])
    
    return H, ansatze, N_op, Sz_op

@pytest.mark.parametrize("ansatz_name", ["DexcG", "UCCSD", "k-UpCCGSD", "PCU2"])
def test_particle_number_and_sz_conservation_optimized(setup_system, ansatz_name):
    H, ansatze, N_op, Sz_op = setup_system
    qc = ansatze[ansatz_name]
    
    # Optimize using GD from zero initialization
    res = run_vqe_single(qc, H, ansatz_name, "zero", "GD", maxiter=20, lr=0.1)
    
    bound_qc = qc.assign_parameters(res["best_params"])
    sv = Statevector(bound_qc)
    
    exp_n = float(sv.expectation_value(N_op).real)
    exp_sz = float(sv.expectation_value(Sz_op).real)
    
    assert np.isclose(exp_n, 2.0, atol=1e-4), f"[{ansatz_name}] Optimized state particle number N = {exp_n:.6f} != 2.0"
    assert np.isclose(exp_sz, 0.0, atol=1e-4), f"[{ansatz_name}] Optimized state Sz = {exp_sz:.6f} != 0.0"

