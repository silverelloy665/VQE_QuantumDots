"""
Verification Test: Reproducibility across identical seeds.
Verifies that rerunning a stochastic or deterministic optimizer configuration
with the same random seed yields identical energy trajectories and final parameters.
"""
import numpy as np
from src.molecule import load_or_build_bn_dot_hamiltonian
from src.ansatze import get_ansatz_dict
from src.optimizers import run_vqe_single

def test_reproducibility_identical_seed():
    H, E_exact, meta = load_or_build_bn_dot_hamiltonian("6-31g(d,p)")
    ansatze = get_ansatz_dict()
    qc = ansatze["UCCSD"]
    
    # Run 1
    res1 = run_vqe_single(
        circuit=qc,
        hamiltonian=H,
        ansatz_name="UCCSD",
        init_name="random",
        optimizer_name="SPSA",
        maxiter=15,
        seed=12345
    )
    
    # Run 2 (identical seed)
    res2 = run_vqe_single(
        circuit=qc,
        hamiltonian=H,
        ansatz_name="UCCSD",
        init_name="random",
        optimizer_name="SPSA",
        maxiter=15,
        seed=12345
    )
    
    assert np.isclose(res1["final_energy"], res2["final_energy"], atol=1e-12), "Reproducibility failed: final energy differs"
    assert np.allclose(res1["final_params"], res2["final_params"], atol=1e-12), "Reproducibility failed: final parameters differ"
    assert np.allclose(res1["energy_history"], res2["energy_history"], atol=1e-12), "Reproducibility failed: history differs"

