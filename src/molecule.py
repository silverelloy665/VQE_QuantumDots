"""
Molecular geometry and active-space electronic Hamiltonian builder for the
hexagonal Boron-Nitride (B8N8H10) quantum dot using Jordan-Wigner transformation.
"""
import json
import warnings
from pathlib import Path
import numpy as np
from scipy.sparse import SparseEfficiencyWarning

from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.mappers import JordanWignerMapper

warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

GEOMETRY_STR = """
B -3.38352416 3.02033458 0.0
N -4.65065508 2.33624451 0.0
N -2.15751932 2.26501205 0.0
B -4.69178116 0.89683190 0.0
B -2.19864540 0.82559945 0.0
H -3.34953803 4.20984917 0.0
H -5.51056126 2.86601934 0.0
H -1.26876777 2.74482523 0.0
N -3.46577632 0.14150937 0.0
B -3.50690239 -1.29790323 0.0
B -6.00003816 -1.22667078 0.0
N -5.95891208 0.21274183 0.0
N -4.77403331 -1.98199331 0.0
N -0.97264055 0.07027692 0.0
N -2.28089755 -2.05322576 0.0
B -1.01376663 -1.36913568 0.0
H -6.81881825 0.74251666 0.0
H -7.04718107 -1.79199522 0.0
H 0.07450236 0.63560136 0.0
H -0.00060985 -1.99332583 0.0
B -4.81515939 -3.42140591 0.0
B -2.32202362 -3.49263836 0.0
N -3.58915454 -4.17672844 0.0
H -3.61799991 -5.18631645 0.0
H -5.86230230 -3.98673035 0.0
H -1.30886684 -4.11682851 0.0
""".strip()

def load_or_build_bn_dot_hamiltonian(
    basis: str = "6-31g(d,p)",
    n_active_electrons: int = 2,
    n_active_orbitals: int = 2,
    orbital_method: str = "rhf"
) -> tuple[SparsePauliOp, float, dict]:
    """
    Loads active-space integrals for BN Quantum Dot and builds the Qiskit Pauli Hamiltonian.
    
    Returns:
        qubit_hamiltonian (SparsePauliOp): Qubit Hamiltonian including constant energy shift.
        exact_energy (float): Exact ground state eigenvalue in Hartree.
        meta (dict): Dictionary with molecule metadata (electrons, active space, etc.)
    """
    clean_basis = "631gdp" if "6-31g" in basis.lower() else "sto3g"
    
    if orbital_method.lower() == "b3lyp":
        cache_filename = f"bn_dot_{clean_basis}_b3lyp.json"
    elif (n_active_electrons, n_active_orbitals) == (4, 4):
        cache_filename = f"bn_dot_{clean_basis}_4e4o.json"
    elif (n_active_electrons, n_active_orbitals) == (6, 6):
        cache_filename = f"bn_dot_{clean_basis}_6e6o.json"
    else:
        cache_filename = f"bn_dot_{clean_basis}.json"

    cache_path = DATA_DIR / cache_filename

    if not cache_path.exists():
        raise FileNotFoundError(
            f"Active space cache file not found at {cache_path}. "
            f"Run 'wsl python3 src/compute_pyscf.py' to generate active space data."
        )

    with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    h1 = np.array(data["h1_active"])
    h2 = np.array(data["h2_active"])
    
    # Convert to Physicist notation <pr|qs> = (pq|rs) -> np.einsum('pqrs->prqs', h2)
    h2_phys = np.einsum("pqrs->prqs", h2)
    ee = ElectronicEnergy.from_raw_integrals(h1, h2_phys)
    fermionic_op = ee.second_q_op()
    
    mapper = JordanWignerMapper()
    active_qubit_op = mapper.map(fermionic_op)
    
    # Exact diagonalization of active space Hamiltonian
    mat = active_qubit_op.to_matrix()
    evals, _ = np.linalg.eigh(mat)
    active_ground_energy = float(evals[0])
    
    casci_energy = float(data["casci_energy"])
    hf_energy = float(data["hf_energy"])
    energy_shift = casci_energy - active_ground_energy
    
    # Add identity term shift to qubit Hamiltonian
    num_qubits = active_qubit_op.num_qubits
    id_op = SparsePauliOp(["I" * num_qubits], [energy_shift])
    full_qubit_hamiltonian = (active_qubit_op + id_op).simplify()
    
    e_corr_mHa = abs(hf_energy - casci_energy) * 1000.0
    if e_corr_mHa < 1.0:
        warnings.warn(
            f"Active space ({n_active_electrons}e, {n_active_orbitals}o) has small correlation energy "
            f"E_corr = {e_corr_mHa:.4f} mHa (< 1.0 mHa). Zero-initialization starts within {e_corr_mHa:.4f} mHa "
            f"of exact reference.",
            UserWarning
        )
    
    meta = {
        "molecule": data["molecule"],
        "label": data["label"],
        "basis": data["basis"],
        "orbital_method": data.get("orbital_method", orbital_method),
        "n_atoms": data["n_atoms"],
        "n_electrons": data["n_electrons"],
        "n_active_electrons": data["n_active_electrons"],
        "n_active_orbitals": data["n_active_orbitals"],
        "num_qubits": num_qubits,
        "num_pauli_terms": len(full_qubit_hamiltonian),
        "hf_energy": hf_energy,
        "casci_energy": casci_energy,
        "active_ground_energy": active_ground_energy,
        "energy_shift": energy_shift,
        "exact_ground_energy": casci_energy,
        "correlation_energy_mHa": e_corr_mHa,
        "dft_total_energy": data.get("dft_total_energy")
    }
    
    return full_qubit_hamiltonian, casci_energy, meta

if __name__ == "__main__":
    for b in ["sto-3g", "6-31g(d,p)"]:
        H, E_exact, meta = load_or_build_bn_dot_hamiltonian(basis=b)
        print(f"[{meta['label']} - {b}] Qubits: {meta['num_qubits']}, Pauli terms: {meta['num_pauli_terms']}, Exact Energy: {E_exact:.8f} Ha, E_corr: {meta['correlation_energy_mHa']:.4f} mHa")
