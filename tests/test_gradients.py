"""
Verification Test: Gradient computation accuracy and GD monotonicity on DexcG.
Verifies:
1. Batched central finite difference gradient vs independent finite differences (< 1e-4)
   at zero, half, and 3 random points across all 4 ansatze.
2. Monotonic energy descent of GD on DexcG from a non-zero initial point.
"""
import numpy as np
import pytest
from qiskit.primitives import StatevectorEstimator
from src.molecule import load_or_build_bn_dot_hamiltonian
from src.ansatze import get_ansatz_dict
from src.optimizers import compute_gradient_batched, run_gradient_descent

@pytest.fixture(scope="module")
def setup_hamiltonian_and_ansatze():
    H, E_exact, meta = load_or_build_bn_dot_hamiltonian("6-31g(d,p)")
    ansatze = get_ansatz_dict()
    estimator = StatevectorEstimator()
    return H, E_exact, meta, ansatze, estimator

def test_gradient_accuracy_across_ansatze(setup_hamiltonian_and_ansatze):
    H, E_exact, meta, ansatze, estimator = setup_hamiltonian_and_ansatze
    
    # Test points: 5 fixed random seeds for each ansatz
    seeds = [0, 1, 2, 3, 42]
    eps = 1e-5
    
    for name, qc in ansatze.items():
        n_params = qc.num_parameters
        for s in seeds:
            rng = np.random.RandomState(s)
            pt = rng.uniform(0.0, 2.0 * np.pi, n_params)
            
            # Batched gradient (default: finite_diff)
            g_batched = compute_gradient_batched(qc, H, pt, estimator, method="finite_diff", eps=eps)
            
            # Independent single-point finite difference calculation
            g_indep = np.zeros(n_params)
            for i in range(n_params):
                p_plus = np.copy(pt)
                p_minus = np.copy(pt)
                p_plus[i] += eps
                p_minus[i] -= eps
                
                e_plus = float(estimator.run([(qc, H, [p_plus])]).result()[0].data.evs[0])
                e_minus = float(estimator.run([(qc, H, [p_minus])]).result()[0].data.evs[0])
                g_indep[i] = (e_plus - e_minus) / (2.0 * eps)
                
            max_diff = np.max(np.abs(g_batched - g_indep))
            assert max_diff < 1e-6, f"Gradient mismatch for {name} at seed {s}: max diff = {max_diff:.2e}"

def test_dexg_gradient_at_zero(setup_hamiltonian_and_ansatze):
    H, E_exact, meta, ansatze, estimator = setup_hamiltonian_and_ansatze
    qc = ansatze["DexcG"]
    th0 = np.zeros(qc.num_parameters)
    g = compute_gradient_batched(qc, H, th0, estimator, method="finite_diff", eps=1e-5)
    assert abs(g[0] - 0.0126) < 1e-4, f"DexcG gradient at 0 is {g[0]:.6f}, expected 0.0126 +/- 1e-4"

def test_param_shift_pi4_dexcg(setup_hamiltonian_and_ansatze):
    H, E_exact, meta, ansatze, estimator = setup_hamiltonian_and_ansatze
    qc = ansatze["DexcG"]
    eps = 1e-5
    for s in [0, 1, 2, 3, 42]:
        rng = np.random.RandomState(s)
        pt = rng.uniform(0.0, 2.0 * np.pi, qc.num_parameters)
        g_fd = compute_gradient_batched(qc, H, pt, estimator, method="finite_diff", eps=eps)
        g_ps = compute_gradient_batched(qc, H, pt, estimator, method="param_shift_pi4")
        assert np.max(np.abs(g_fd - g_ps)) < 1e-6, f"param_shift_pi4 mismatch on DexcG at seed {s}: {np.max(np.abs(g_fd - g_ps)):.2e}"

def test_gradient_descent_monotonic_decrease_dexcg(setup_hamiltonian_and_ansatze):
    H, E_exact, meta, ansatze, estimator = setup_hamiltonian_and_ansatze
    qc = ansatze["DexcG"]
    
    init_params = np.full(qc.num_parameters, 0.5)
    # Run GD with small conservative learning rate (e.g. lr=0.05)
    final_e, final_p, hist, wtime, extra = run_gradient_descent(
        circuit=qc,
        hamiltonian=H,
        init_params=init_params,
        estimator=estimator,
        maxiter=30,
        lr=0.05,
        momentum=0.0
    )
    
    # Verify strict monotonic descent: hist[i+1] <= hist[i] + 1e-12
    for i in range(len(hist) - 1):
        assert hist[i+1] <= hist[i] + 1e-10, f"Monotonic decrease violated at step {i}: {hist[i]} -> {hist[i+1]}"
        
    assert final_e < hist[0], "GD failed to decrease energy from initial point"


def test_hard_problem_gradient_accuracy():
    """
    Assert gradient correctness vs central finite difference for each ansatz
    on the hard benchmark problem (8 qubits) at 5 random points (tolerance 1e-6)
    as required by Phase C2.
    """
    from src.molecule import load_hard_hamiltonian
    from src.optimizers import FastStatevectorEstimator
    H, _, _ = load_hard_hamiltonian()
    ansatze = get_ansatz_dict(4, (2, 2), include_k_variants=True)
    estimator = FastStatevectorEstimator(H)
    seeds = [0, 1, 2, 3, 42]
    eps = 1e-5

    for name in ["DexcG", "UCCSD", "UCCSD-SA", "1-UpCCGSD"]:
        qc = ansatze[name].decompose().decompose()
        n_params = qc.num_parameters
        for s in seeds:
            rng = np.random.RandomState(s)
            pt = rng.uniform(0.0, 2.0 * np.pi, n_params)
            g_batched = compute_gradient_batched(qc, H, pt, estimator, method="finite_diff", eps=eps)

            # Independent central finite difference
            g_indep = np.zeros(n_params)
            for i in range(n_params):
                p_plus = np.copy(pt)
                p_minus = np.copy(pt)
                p_plus[i] += eps
                p_minus[i] -= eps
                e_plus = float(estimator.run([(qc, H, [p_plus])]).result()[0].data.evs[0])
                e_minus = float(estimator.run([(qc, H, [p_minus])]).result()[0].data.evs[0])
                g_indep[i] = (e_plus - e_minus) / (2.0 * eps)

            max_diff = np.max(np.abs(g_batched - g_indep))
            assert max_diff < 1e-6, (
                f"Gradient mismatch for {name} on hard problem at seed {s}: max diff = {max_diff:.2e}"
            )


