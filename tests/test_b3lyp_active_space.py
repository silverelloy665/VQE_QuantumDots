"""
Verification Test: Active space calculation in B3LYP orbitals.
Verifies:
1. Active space CASCI can be computed using Kohn-Sham DFT (B3LYP) molecular orbitals.
2. Clearly distinguishes between the active-space energy in B3LYP orbitals (-639.638 Ha)
   and the full B3LYP DFT total electronic energy (-643.634 Ha).
3. VQE executes successfully on the B3LYP-orbital Hamiltonian.
"""
from pathlib import Path
import numpy as np
import pytest
from src.molecule import load_or_build_bn_dot_hamiltonian
from src.ansatze import get_ansatz_dict
from src.optimizers import run_vqe_single

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def test_b3lyp_orbitals_active_space_vqe():
    cache_path = DATA_DIR / "bn_dot_631gdp_b3lyp.json"
    if not cache_path.exists():
        pytest.skip("B3LYP active space cache not yet generated")
        
    H_b3lyp, E_exact_b3lyp, meta_b3lyp = load_or_build_bn_dot_hamiltonian(
        basis="6-31g(d,p)", n_active_electrons=2, n_active_orbitals=2, orbital_method="b3lyp"
    )
    
    # Check that DFT total energy is separate and distinct
    dft_total = meta_b3lyp.get("dft_total_energy")
    assert dft_total is not None, "dft_total_energy not recorded in metadata"
    assert dft_total < -643.0, f"Expected total B3LYP energy ~ -643.63 Ha, got {dft_total}"
    assert E_exact_b3lyp > -640.0, f"Expected active space energy ~ -639.64 Ha, got {E_exact_b3lyp}"
    
    # Run VQE with DexcG
    ansatze = get_ansatz_dict()
    qc = ansatze["DexcG"]
    res = run_vqe_single(qc, H_b3lyp, "DexcG", "zero", "GD", maxiter=25, lr=0.1)
    
    # VQE should converge close to active-space exact energy
    err_mHa = abs(res["final_energy"] - E_exact_b3lyp) * 1000.0
    assert err_mHa < 1.0, f"VQE on B3LYP orbitals failed to reach active-space energy: err = {err_mHa:.4f} mHa"

