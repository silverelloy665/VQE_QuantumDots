"""
Tests for the genuinely hard benchmark Hamiltonian (BN Quantum Dot, B8N8H10).
Verifies:
- Active space correlation energy >= 5.0 mHa.
- Number of qubits <= 12 (8 qubits for 4e, 4o).
- Diagonalization of the full 256-dimensional Fock space matches CASCI to < 1e-8 Ha.
- Unconstrained Fock-space ground state lies strictly in the (N_alpha=2, N_beta=2) sector.
"""
import json
from pathlib import Path
import numpy as np
import pytest

from src.molecule import load_hard_hamiltonian, DATA_DIR


def test_hard_hamiltonian_metadata_and_gap():
    """Verify metadata and correlation energy >= 5.0 mHa on bn_dot_hard.json."""
    hard_path = DATA_DIR / "bn_dot_hard.json"
    assert hard_path.exists(), "bn_dot_hard.json must exist in data directory"

    with open(hard_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["molecule"] == "B8N8H10"
    assert data["n_active_electrons"] == 4
    assert data["n_active_orbitals"] == 4
    assert data["num_qubits"] == 8
    assert data["scale"] == 1.4
    assert data["orbital_selection"] == "natural_orbitals"

    e_ref = float(data["reference_determinant_energy"])
    e_casci = float(data["casci_energy"])
    gap_mha = (e_ref - e_casci) * 1000.0

    assert gap_mha >= 5.0, f"Correlation energy {gap_mha:.4f} mHa must be >= 5.0 mHa"
    assert abs(gap_mha - float(data["correlation_energy_mHa"])) < 1e-3


def test_hard_hamiltonian_full_fock_space_diagonalization():
    """
    Diagonalize the 8-qubit (2^8 = 256) Hamiltonian in full Fock space without particle constraints.
    Assert:
    1. Lowest eigenvalue matches CASCI to < 1e-8 Ha.
    2. Ground state lies strictly in the (N_alpha=2, N_beta=2) sector with weight > 0.9999.
    """
    H, E_exact, meta = load_hard_hamiltonian()

    assert meta["num_qubits"] == 8
    assert len(H) > 0

    mat = H.to_matrix()
    evals, evecs = np.linalg.eigh(mat)

    # Lowest eigenvalue in full Fock space
    fock_ground_energy = float(evals[0])
    casci_energy = meta["casci_energy"]

    err = abs(fock_ground_energy - casci_energy)
    assert err < 1e-8, (
        f"Full Fock space ground energy {fock_ground_energy:.10f} Ha differs from "
        f"CASCI reference {casci_energy:.10f} Ha by {err:.2e} Ha (must be < 1e-8 Ha)"
    )

    # Check particle sector of ground state
    # Jordan-Wigner ordering: first 4 qubits are alpha, last 4 qubits are beta
    gs = evecs[:, 0]
    n_alpha_target = meta["n_active_electrons"] // 2  # 2
    n_beta_target = meta["n_active_electrons"] // 2   # 2

    sector_weights = {}
    for idx in range(len(gs)):
        prob = abs(gs[idx]) ** 2
        if prob > 1e-8:
            bits = format(idx, "08b")
            q_occ = [int(b) for b in reversed(bits)]
            n_alpha = sum(q_occ[:4])
            n_beta = sum(q_occ[4:])
            sec = (n_alpha, n_beta)
            sector_weights[sec] = sector_weights.get(sec, 0.0) + prob

    target_sec = (n_alpha_target, n_beta_target)
    target_weight = sector_weights.get(target_sec, 0.0)

    assert target_weight > 0.9999, (
        f"Ground state weight in target sector {target_sec} is {target_weight:.6f} (< 0.9999). "
        f"Sector breakdown: {sector_weights}"
    )

