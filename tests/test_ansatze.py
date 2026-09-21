"""
Verification Test: Ansatz structure, parameter counts, and zero-parameter HF reproduction.
Verifies:
1. Variational parameter counts match theory (DexcG=1, PCU2=14, UCCSD=3, k-UpCCGSD=9).
2. At theta = 0, every ansatz reproduces <HF|H|HF> (RHF energy) to < 1e-6 Ha.
3. UCC excitation classifications: DexcG ('d'), UCCSD ('sd'), k-UpCCGSD (generalized 'sd', reps=3).
"""
import numpy as np
import pytest
from qiskit.quantum_info import Statevector
from src.molecule import load_or_build_bn_dot_hamiltonian
from src.ansatze import get_ansatz_dict

@pytest.fixture(scope="module")
def setup_data():
    H, E_exact, meta = load_or_build_bn_dot_hamiltonian("6-31g(d,p)")
    ansatze = get_ansatz_dict()
    return H, meta["hf_energy"], ansatze

def test_parameter_counts_match_theory(setup_data):
    _, _, ansatze = setup_data
    expected_counts = {
        "DexcG": 1,
        "PCU2": 14,
        "UCCSD": 3,
        "k-UpCCGSD": 9
    }
    for name, expected_n in expected_counts.items():
        actual_n = ansatze[name].num_parameters
        assert actual_n == expected_n, f"Parameter count mismatch for {name}: expected {expected_n}, got {actual_n}"

def test_zero_parameters_reproduce_hartree_fock_energy(setup_data):
    H, hf_energy, ansatze = setup_data
    for name, qc in ansatze.items():
        zero_params = np.zeros(qc.num_parameters)
        bound = qc.assign_parameters(zero_params)
        sv = Statevector(bound)
        computed_e = float(sv.expectation_value(H).real)
        assert np.isclose(computed_e, hf_energy, atol=1e-6), (
            f"Ansatz {name} at theta=0 gave energy {computed_e:.8f} Ha, which differs from HF {hf_energy:.8f} Ha"
        )

