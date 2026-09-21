import json
import numpy as np
from pyscf import gto, scf, mcscf, ao2mo

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

def compute_active_space(basis="sto-3g", n_active_electrons=2, n_active_orbitals=2):
    mol = gto.Mole()
    mol.atom = GEOMETRY_STR
    mol.basis = basis
    mol.charge = 0
    mol.spin = 0
    mol.unit = "Angstrom"
    mol.build()

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
    h1e_active, e_core = cas.get_h1eff(mo_coeff=mo_coeff)
    e_inactive = float(e_core) + nuclear_repulsion_energy
    
    # 2-electron integrals in active space (chemist notation: (pq|rs))
    cas_mo = mo_coeff[:, cas_idx]
    eri_active = ao2mo.kernel(mol, cas_mo)
    eri_active = ao2mo.restore(1, eri_active, ncas) # 4-index tensor (ncas, ncas, ncas, ncas)

    # CASCI reference energy for active space
    mc = mcscf.CASCI(mf, n_active_orbitals, n_active_electrons)
    e_casci = float(mc.kernel()[0])
    print(f"[{basis.upper()}] CASCI Ground State Energy = {e_casci:.8f} Ha")

    data = {
        "molecule": "B8N8H10",
        "label": "BN quantum dot",
        "basis": basis,
        "n_atoms": 26,
        "n_electrons": mol.nelectron,
        "n_active_electrons": n_active_electrons,
        "n_active_orbitals": n_active_orbitals,
        "num_qubits": n_active_orbitals * 2,
        "hf_energy": hf_energy,
        "nuclear_repulsion_energy": nuclear_repulsion_energy,
        "inactive_energy": float(e_core),
        "total_inactive_energy": e_inactive,
        "casci_energy": e_casci,
        "h1_active": h1e_active.tolist(),
        "h2_active": eri_active.tolist(),
        "mo_energies": mf.mo_energy.tolist()[:10],
    }
    return data

if __name__ == "__main__":
    sto3g_data = compute_active_space(basis="sto-3g", n_active_electrons=2, n_active_orbitals=2)
    with open("data/bn_dot_sto3g.json", "w") as f:
        json.dump(sto3g_data, f, indent=2)
    print("[OK] Saved STO-3G data to data/bn_dot_sto3g.json")

    basis_631gdp_data = compute_active_space(basis="6-31g(d,p)", n_active_electrons=2, n_active_orbitals=2)
    with open("data/bn_dot_631gdp.json", "w") as f:
        json.dump(basis_631gdp_data, f, indent=2)
    print("[OK] Saved 6-31G(d,p) data to data/bn_dot_631gdp.json")

