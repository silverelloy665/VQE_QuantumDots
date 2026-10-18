"""
Verification Test: Hamiltonian properties and sector exact diagonalization.
Verifies:
1. <HF|H|HF> equals PySCF RHF energy to < 1e-6 Ha.
2. Exact diagonalization restricted to the N=2, Sz=0 sector equals CASCI ground energy.
"""
import numpy as np
import pytest
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_nature.second_q.mappers import JordanWignerMapper
from src.molecule import load_or_build_bn_dot_hamiltonian

@pytest.mark.parametrize("basis, expected_rhf, expected_casci", [
    ("sto-3g", -631.74167448, -631.74178346),
    ("6-31g(d,p)", -639.72424230, -639.72428323)
])
def test_hf_energy_and_sector_diagonalization(basis, expected_rhf, expected_casci):
    H, E_casci, meta = load_or_build_bn_dot_hamiltonian(basis=basis)
    mapper = JordanWignerMapper()
    
    # 1. <HF|H|HF>
    hf_circuit = HartreeFock(2, (1, 1), mapper)
    sv_hf = Statevector(hf_circuit)
    computed_hf_e = float(sv_hf.expectation_value(H).real)
    
    assert np.isclose(computed_hf_e, expected_rhf, atol=1e-6), (
        f"[{basis}] <HF|H|HF> = {computed_hf_e:.8f} deviates from RHF = {expected_rhf:.8f}"
    )
    assert np.isclose(computed_hf_e, meta["hf_energy"], atol=1e-6), (
        f"[{basis}] <HF|H|HF> does not match cached RHF energy"
    )
    
    # 2. Sector restriction: N=2, Sz=0 subspace exact diagonalization
    # 4 qubits -> 16 basis states. N=2, Sz=0 states require:
    # 1 alpha electron on qubit {0,1} and 1 beta electron on qubit {2,3}.
    # The 4 valid configurations are |q3 q2 q1 q0>:
    # |0101> (HF), |1001>, |0110>, |1010>
    h_mat = H.to_matrix()
    
    # Identify basis state indices with N_alpha=1 and N_beta=1
    valid_indices = []
    for idx in range(16):
        b = f"{idx:04b}" # q3, q2, q1, q0
        q0 = int(b[3])
        q1 = int(b[2])
        q2 = int(b[1])
        q3 = int(b[0])
        n_alpha = q0 + q1
        n_beta = q2 + q3
        if n_alpha == 1 and n_beta == 1:
            valid_indices.append(idx)
            
    assert len(valid_indices) == 4, f"Expected 4 states in N=2, Sz=0 sector, found {len(valid_indices)}"
    
    sector_mat = h_mat[np.ix_(valid_indices, valid_indices)]
    evals, _ = np.linalg.eigh(sector_mat)
    sector_ground_e = float(evals[0].real)
    
    assert np.isclose(sector_ground_e, expected_casci, atol=1e-6), (
        f"[{basis}] Sector ground energy {sector_ground_e:.8f} Ha deviates from CASCI {expected_casci:.8f} Ha"
    )

