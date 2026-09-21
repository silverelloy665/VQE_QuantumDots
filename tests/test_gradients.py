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
    
    # Test points: zero, half, and 3 fixed random seeds
    seeds = [42, 123, 999]
    eps = 1e-5
    
    for name, qc in ansatze.items():
        n_params = qc.num_parameters
        test_points = [
            ("zero", np.zeros(n_params)),
            ("half", np.full(n_params, 0.5)),
        ]
        for s in seeds:
            rng = np.random.RandomState(s)
            test_points.append((f"random_seed_{s}", rng.uniform(0.0, 1.0, n_params)))
            
        for pt_name, pt in test_points:
            # Batched gradient
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
            assert max_diff < 1e-4, f"Gradient mismatch for {name} at {pt_name}: max diff = {max_diff:.2e}"

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

