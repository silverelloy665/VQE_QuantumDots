"""
Verification Test: UCCSD Circuit Invariance vs Basis Set and Analytic Scaling.
Verifies:
1. UCCSD for identical active space (2e, 2o) built for STO-3G and 6-31G(d,p)
   has identical num_qubits, num_parameters, and gate counts.
2. The active-space Hamiltonians for STO-3G and 6-31G(d,p) differ.
3. UCCSD parameter counts computed from the circuit match analytic scaling for (2e,2o), (4e,4o), (6e,6o).
4. UCCSD at theta=0 reproduces the reference determinant energy in both STO-3G and 6-31G(d,p).
"""
import numpy as np
import pytest
from qiskit.quantum_info import Statevector
from qiskit_nature.second_q.circuit.library import UCC, HartreeFock
from qiskit_nature.second_q.mappers import JordanWignerMapper

from src.molecule import load_or_build_bn_dot_hamiltonian
from src.scaling import compute_uccsd_counts


def test_uccsd_circuit_invariance_and_hamiltonian_difference():
    # 1. Load Hamiltonians for STO-3G and 6-31G(d,p)
    H_sto, _, meta_sto = load_or_build_bn_dot_hamiltonian("sto-3g", 2, 2)
    H_631, _, meta_631 = load_or_build_bn_dot_hamiltonian("6-31g(d,p)", 2, 2)

    # (ii) Assert the two Hamiltonians differ
    diff_op = (H_sto - H_631).simplify()
    assert len(diff_op) > 0, "Expected STO-3G and 6-31G(d,p) Hamiltonians to differ"
    assert not np.allclose(H_sto.coeffs, H_631.coeffs), "Hamiltonian coefficients must differ between bases"

    # Build UCCSD for both
    mapper = JordanWignerMapper()
    hf_sto = HartreeFock(2, (1, 1), mapper)
    hf_631 = HartreeFock(2, (1, 1), mapper)

    ucc_sto = UCC(2, (1, 1), excitations="sd", qubit_mapper=mapper, initial_state=hf_sto)
    ucc_631 = UCC(2, (1, 1), excitations="sd", qubit_mapper=mapper, initial_state=hf_631)

    # (i) Assert identical num_qubits, num_parameters, and gate counts
    assert ucc_sto.num_qubits == ucc_631.num_qubits == 4
    assert ucc_sto.num_parameters == ucc_631.num_parameters == 3

    decomp_sto = ucc_sto.decompose().decompose()
    decomp_631 = ucc_631.decompose().decompose()
    assert decomp_sto.count_ops() == decomp_631.count_ops()
    assert decomp_sto.depth() == decomp_631.depth()

    # (iv) Assert UCCSD at theta=0 gives reference determinant energy in both bases
    theta_0 = np.zeros(3)
    sv_sto = Statevector(ucc_sto.assign_parameters(theta_0))
    e_sto = float(sv_sto.expectation_value(H_sto).real)
    assert np.isclose(e_sto, meta_sto["reference_determinant_energy"], atol=1e-6), (
        f"STO-3G UCCSD at theta=0 gave {e_sto}, expected {meta_sto['reference_determinant_energy']}"
    )

    sv_631 = Statevector(ucc_631.assign_parameters(theta_0))
    e_631 = float(sv_631.expectation_value(H_631).real)
    assert np.isclose(e_631, meta_631["reference_determinant_energy"], atol=1e-6), (
        f"6-31G(d,p) UCCSD at theta=0 gave {e_631}, expected {meta_631['reference_determinant_energy']}"
    )


@pytest.mark.parametrize("n_e, n_o", [(2, 2), (4, 4), (6, 6)])
def test_uccsd_parameter_counts_match_analytic(n_e, n_o):
    # (iii) Assert UCCSD parameter counts computed from circuit equal analytic counts
    mapper = JordanWignerMapper()
    hf = HartreeFock(n_o, (n_e // 2, n_e // 2), mapper)
    uccsd = UCC(n_o, (n_e // 2, n_e // 2), excitations="sd", qubit_mapper=mapper, initial_state=hf)

    analytic = compute_uccsd_counts(n_e, n_o)
    assert uccsd.num_parameters == analytic["total_uccsd"], (
        f"UCCSD ({n_e}e, {n_o}o) parameter count {uccsd.num_parameters} != analytic {analytic['total_uccsd']}"
    )

