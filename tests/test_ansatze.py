"""
Verification Test: Ansatz structure, parameter counts, and zero-parameter reference determinant reproduction.
Verifies:
1. Variational parameter counts match theory (DexcG=1, PCU2=14, UCCSD=3, k-UpCCGSD=9).
2. At theta = 0, every ansatz reproduces <HF|H|HF> (reference determinant energy) to < 1e-6 Ha for RHF orbitals.
3. At theta = 0, every ansatz reproduces the HF determinant in B3LYP orbitals to < 1e-6 Ha.
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
    return H, meta["reference_determinant_energy"], ansatze

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

def test_zero_parameters_reproduce_reference_determinant_energy(setup_data):
    H, ref_det_energy, ansatze = setup_data
    for name, qc in ansatze.items():
        zero_params = np.zeros(qc.num_parameters)
        bound = qc.assign_parameters(zero_params)
        sv = Statevector(bound)
        computed_e = float(sv.expectation_value(H).real)
        assert np.isclose(computed_e, ref_det_energy, atol=1e-6), (
            f"Ansatz {name} at theta=0 gave energy {computed_e:.8f} Ha, which differs from reference {ref_det_energy:.8f} Ha"
        )

def test_zero_parameters_reproduce_b3lyp_reference_determinant():
    H_b3lyp, _, meta_b3lyp = load_or_build_bn_dot_hamiltonian(
        basis="6-31g(d,p)", n_active_electrons=2, n_active_orbitals=2, orbital_method="b3lyp"
    )
    ref_det_energy = meta_b3lyp["reference_determinant_energy"]
    ansatze = get_ansatz_dict()
    for name, qc in ansatze.items():
        zero_params = np.zeros(qc.num_parameters)
        bound = qc.assign_parameters(zero_params)
        sv = Statevector(bound)
        computed_e = float(sv.expectation_value(H_b3lyp).real)
        assert np.isclose(computed_e, ref_det_energy, atol=1e-6), (
            f"Ansatz {name} at theta=0 on B3LYP gave energy {computed_e:.8f} Ha, "
            f"which differs from reference {ref_det_energy:.8f} Ha"
        )
