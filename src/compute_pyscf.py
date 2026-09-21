"""
Electronic structure calculations for BN Quantum Dot (B8N8H10) using PySCF.
Computes RHF, DFT/B3LYP, and CASCI active spaces ((2e,2o), (4e,4o), (6e,6o)).
"""
import os
import sys
import json
import numpy as np
from pyscf import gto, scf, mcscf, ao2mo
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

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
"""
""".strip()

def compute_active_space(basis="sto-3g", n_active_electrons=2, n_active_orbitals=2):
def compute_active_space(
    basis: str = "6-31g(d,p)",
    n_active_electrons: int = 2,
    n_active_orbitals: int = 2,
    orbital_method: str = "rhf"
) -> dict:
    """
    Performs electronic structure calculation for B8N8H10 and generates active-space integrals.
    Requires PySCF (run in PySCF-supported environment e.g. Linux / WSL).
    """
    from pyscf import gto, scf, mcscf, ao2mo, dft
    
    mol = gto.Mole()
    mol.atom = GEOMETRY_STR
    mol.basis = basis
    mol.charge = 0
    mol.spin = 0
    mol.unit = "Angstrom"
    mol.build()
    
    nuclear_repulsion = float(mol.energy_nuc())
    
    if orbital_method.lower() == "b3lyp":
        print(f"[{basis.upper()}] Computing RKS (B3LYP)...")
        mf = dft.RKS(mol)
        mf.xc = "b3lyp"
        mf.kernel()
        scf_energy = float(mf.e_tot)
        dft_total_energy = scf_energy
        print(f"[{basis.upper()}] B3LYP total energy = {dft_total_energy:.8f} Ha")
    else:
        print(f"[{basis.upper()}] Computing RHF...")
        mf = scf.RHF(mol)
        mf.kernel()
        scf_energy = float(mf.e_tot)
        dft_total_energy = None
        print(f"[{basis.upper()}] RHF energy = {scf_energy:.8f} Ha")

    print(f"[{basis.upper()}] Computing RHF...")
    mf = scf.RHF(mol)
    mf.kernel()
    hf_energy = float(mf.e_tot)
    nuclear_repulsion_energy = float(mol.energy_nuc())
    print(f"[{basis.upper()}] RHF Energy = {hf_energy:.8f} Ha, Nuc = {nuclear_repulsion_energy:.8f} Ha")

    # CAS active space calculation
    cas = mcscf.CASSCF(mf, n_active_orbitals, n_active_electrons)
    # Compute 1-e and 2-e integrals in active space
    mo_coeff = mf.mo_coeff
    ncore = (mol.nelectron - n_active_electrons) // 2
    ncas = n_active_orbitals
    cas_idx = slice(ncore, ncore + ncas)
    
    # Active space core energy and effective 1-e integrals
    cas = mcscf.CASSCF(mf, n_active_orbitals, n_active_electrons)
    h1e_active, e_core = cas.get_h1eff(mo_coeff=mo_coeff)
    e_inactive = float(e_core) + nuclear_repulsion_energy
    e_inactive = float(e_core) + nuclear_repulsion
    
    # 2-electron integrals in active space (chemist notation: (pq|rs))
    cas_mo = mo_coeff[:, cas_idx]
    eri_active = ao2mo.kernel(mol, cas_mo)
    eri_active = ao2mo.restore(1, eri_active, ncas) # 4-index tensor (ncas, ncas, ncas, ncas)

    # CASCI reference energy for active space
    eri_active = ao2mo.restore(1, eri_active, ncas) # 4-index tensor
    
    mc = mcscf.CASCI(mf, n_active_orbitals, n_active_electrons)
    e_casci = float(mc.kernel()[0])
    print(f"[{basis.upper()}] CASCI Ground State Energy = {e_casci:.8f} Ha")
    e_corr_mHa = abs(scf_energy - e_casci) * 1000.0
    print(f"[{basis.upper()}] CASCI Ground Energy ({n_active_electrons}e, {n_active_orbitals}o) = {e_casci:.8f} Ha | E_corr = {e_corr_mHa:.5f} mHa")

    data = {
        "molecule": "B8N8H10",
        "label": "BN quantum dot",
        "basis": basis,
        "orbital_method": orbital_method,
        "n_atoms": 26,
        "n_electrons": mol.nelectron,
        "n_active_electrons": n_active_electrons,
        "n_active_orbitals": n_active_orbitals,
        "num_qubits": n_active_orbitals * 2,
        "hf_energy": hf_energy,
        "nuclear_repulsion_energy": nuclear_repulsion_energy,
        "hf_energy": scf_energy if orbital_method == "rhf" else float(scf.RHF(mol).kernel()),
        "scf_energy": scf_energy,
        "dft_total_energy": dft_total_energy,
        "nuclear_repulsion_energy": nuclear_repulsion,
        "inactive_energy": float(e_core),
        "total_inactive_energy": e_inactive,
        "casci_energy": e_casci,
        "correlation_energy_mHa": e_corr_mHa,
        "h1_active": h1e_active.tolist(),
        "h2_active": eri_active.tolist(),
        "mo_energies": mf.mo_energy.tolist()[:10],
    }
    return data

if __name__ == "__main__":
def generate_all_active_space_caches():
    """Generates and saves all active-space caches for STO-3G and 6-31G(d,p)."""
    # 1. STO-3G (2e, 2o)
    sto3g_data = compute_active_space(basis="sto-3g", n_active_electrons=2, n_active_orbitals=2)
    with open("data/bn_dot_sto3g.json", "w") as f:
    with open(DATA_DIR / "bn_dot_sto3g.json", "w") as f:
        json.dump(sto3g_data, f, indent=2)
    print("[OK] Saved STO-3G data to data/bn_dot_sto3g.json")
    print("[OK] Saved bn_dot_sto3g.json")

    basis_631gdp_data = compute_active_space(basis="6-31g(d,p)", n_active_electrons=2, n_active_orbitals=2)
    with open("data/bn_dot_631gdp.json", "w") as f:
        json.dump(basis_631gdp_data, f, indent=2)
    print("[OK] Saved 6-31G(d,p) data to data/bn_dot_631gdp.json")
    # 2. 6-31G(d,p) (2e, 2o)
    gdp_2e2o = compute_active_space(basis="6-31g(d,p)", n_active_electrons=2, n_active_orbitals=2)
    with open(DATA_DIR / "bn_dot_631gdp.json", "w") as f:
        json.dump(gdp_2e2o, f, indent=2)
    print("[OK] Saved bn_dot_631gdp.json")

    # 3. 6-31G(d,p) (4e, 4o)
    gdp_4e4o = compute_active_space(basis="6-31g(d,p)", n_active_electrons=4, n_active_orbitals=4)
    with open(DATA_DIR / "bn_dot_631gdp_4e4o.json", "w") as f:
        json.dump(gdp_4e4o, f, indent=2)
    print("[OK] Saved bn_dot_631gdp_4e4o.json")

    # 4. 6-31G(d,p) (6e, 6o)
    gdp_6e6o = compute_active_space(basis="6-31g(d,p)", n_active_electrons=6, n_active_orbitals=6)
    with open(DATA_DIR / "bn_dot_631gdp_6e6o.json", "w") as f:
        json.dump(gdp_6e6o, f, indent=2)
    print("[OK] Saved bn_dot_631gdp_6e6o.json")

    # 5. 6-31G(d,p) B3LYP orbitals (2e, 2o)
    b3lyp_data = compute_active_space(basis="6-31g(d,p)", n_active_electrons=2, n_active_orbitals=2, orbital_method="b3lyp")
    with open(DATA_DIR / "bn_dot_631gdp_b3lyp.json", "w") as f:
        json.dump(b3lyp_data, f, indent=2)
    print("[OK] Saved bn_dot_631gdp_b3lyp.json")

if __name__ == "__main__":
    generate_all_active_space_caches()
