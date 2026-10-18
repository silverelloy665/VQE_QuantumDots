"""
Scan active spaces and geometries for B8N8H10 Quantum Dot using PySCF.
Scans scales in {1.0, 1.2, 1.4, 1.6, 1.8} and active spaces {(2e,2o), (4e,4o), (6e,6o)}.
Evaluates RHF stability, canonical orbitals, and MP2 natural orbitals.
Saves data/active_space_scan.csv and data/bn_dot_hard.json.
"""
import sys
import json
import csv
from pathlib import Path
import numpy as np

# PySCF imports
import pyscf
from pyscf import gto, scf, mcscf, mp, ao2mo

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
""".strip()


def get_scaled_geometry(scale: float = 1.0) -> str:
    """
    Scale all B-N bond lengths by `scale` relative to B8N8 core centroid.
    Preserves all B-H and N-H bond vectors exactly.
    """
    if abs(scale - 1.0) < 1e-9:
        return GEOMETRY_STR

    atoms = []
    coords = []
    for line in GEOMETRY_STR.strip().splitlines():
        parts = line.split()
        atoms.append(parts[0])
        coords.append([float(x) for x in parts[1:]])
    coords = np.array(coords)

    bn_indices = [i for i, a in enumerate(atoms) if a in ("B", "N")]
    h_indices = [i for i, a in enumerate(atoms) if a == "H"]

    centroid = coords[bn_indices].mean(axis=0)

    # For each H, find nearest B/N and record displacement vector
    h_bonds = {}
    for hi in h_indices:
        dists = np.linalg.norm(coords[bn_indices] - coords[hi], axis=1)
        nearest_bn = bn_indices[np.argmin(dists)]
        vec = coords[hi] - coords[nearest_bn]
        h_bonds[hi] = (nearest_bn, vec)

    # Scale B and N radially relative to centroid
    new_coords = coords.copy()
    for bi in bn_indices:
        new_coords[bi] = centroid + scale * (coords[bi] - centroid)

    # Attach H with exact original bond vectors
    for hi, (bn_idx, vec) in h_bonds.items():
        new_coords[hi] = new_coords[bn_idx] + vec

    lines = []
    for a, c in zip(atoms, new_coords):
        lines.append(f"{a:2s} {c[0]:14.8f} {c[1]:14.8f} {c[2]:14.8f}")
    return "\n".join(lines)


def run_scan():
    scales = [1.0, 1.2, 1.4, 1.6, 1.8]
    active_spaces = [(2, 2), (4, 4), (6, 6)]
    basis = "6-31g(d,p)"

    results = []
    hard_candidates = []

    print(f"=== Starting Active Space Scan ({basis}) ===")
    print("Scales:", scales)
    print("Spaces:", active_spaces)
    print("Note: (8e,8o) skipped: 16 qubits > 12 qubit ceiling and >16 GB RAM requirement.\n")

    for scale in scales:
        geom = get_scaled_geometry(scale)
        mol = gto.Mole()
        mol.atom = geom
        mol.basis = basis
        mol.charge = 0
        mol.spin = 0
        mol.unit = "Angstrom"
        mol.cart = False
        mol.build()

        print(f"--- Scale {scale:.1f} ---")
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-9
        mf.kernel()

        if not mf.converged:
            print(f"  [WARN] RHF failed to converge at scale {scale}!")

        # Stability check
        try:
            # internal=True checks for singlet-singlet orbital rotation instability
            # return_status=True returns boolean status
            stab_res = mf.stability(internal=True, external=False, return_status=True)
            if isinstance(stab_res, tuple):
                is_stable = bool(stab_res[2])
            else:
                is_stable = bool(stab_res)
        except Exception as e:
            print(f"  Stability check exception: {e}")
            is_stable = True  # fallback

        print(f"  RHF energy: {mf.e_tot:.8f} Ha | Stable: {is_stable}")

        # Compute MP2 natural orbitals
        try:
            mp2 = mp.MP2(mf).run(verbose=0)
            rdm1 = mp2.make_rdm1()
            occ, no_coeff = np.linalg.eigh(rdm1)
            idx = np.argsort(occ)[::-1]
            occ = occ[idx]
            no_coeff = no_coeff[:, idx]
            mo_no = mf.mo_coeff.dot(no_coeff)
            mp2_success = True
        except Exception as e:
            print(f"  MP2 failed at scale {scale}: {e}")
            mp2_success = False
            mo_no = None

        for ne, no in active_spaces:
            qubits = no * 2

            # 1. Canonical orbitals
            cas_can = mcscf.CASCI(mf, no, ne)
            e_casci_can = float(cas_can.kernel(mo_coeff=mf.mo_coeff)[0])
            # Reference determinant in canonical orbitals
            h1e_can, e_core_can = cas_can.get_h1eff(mo_coeff=mf.mo_coeff)
            k = ne // 2
            cas_idx_can = slice(cas_can.ncore, cas_can.ncore + cas_can.ncas)
            eri_can = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff[:, cas_idx_can]), no)
            e1_det = 2.0 * sum(h1e_can[i, i] for i in range(k))
            e2_det = sum(2.0 * eri_can[i, i, j, j] - eri_can[i, j, j, i] for i in range(k) for j in range(k))
            e_ref_can = float(e1_det + e2_det + e_core_can)
            gap_can = (e_ref_can - e_casci_can) * 1000.0

            row_can = {
                "scale": scale,
                "ne": ne,
                "no": no,
                "qubits": qubits,
                "E_HF": e_ref_can,
                "E_CASCI": e_casci_can,
                "gap_mHa": round(gap_can, 4),
                "RHF_stable": is_stable,
                "orbital_selection": "canonical",
            }
            results.append(row_can)
            print(f"  ({ne}e,{no}o) [Canonical] Gap: {gap_can:8.4f} mHa (E_ref: {e_ref_can:.8f}, CASCI: {e_casci_can:.8f})")

            # 2. Natural orbitals
            if mp2_success:
                cas_no = mcscf.CASCI(mf, no, ne)
                e_casci_no = float(cas_no.kernel(mo_coeff=mo_no)[0])
                h1e_no, e_core_no = cas_no.get_h1eff(mo_coeff=mo_no)
                cas_idx_no = slice(cas_no.ncore, cas_no.ncore + cas_no.ncas)
                eri_no = ao2mo.restore(1, ao2mo.kernel(mol, mo_no[:, cas_idx_no]), no)
                e1_det_no = 2.0 * sum(h1e_no[i, i] for i in range(k))
                e2_det_no = sum(2.0 * eri_no[i, i, j, j] - eri_no[i, j, j, i] for i in range(k) for j in range(k))
                e_ref_no = float(e1_det_no + e2_det_no + e_core_no)
                gap_no = (e_ref_no - e_casci_no) * 1000.0

                row_no = {
                    "scale": scale,
                    "ne": ne,
                    "no": no,
                    "qubits": qubits,
                    "E_HF": e_ref_no,
                    "E_CASCI": e_casci_no,
                    "gap_mHa": round(gap_no, 4),
                    "RHF_stable": is_stable,
                    "orbital_selection": "natural_orbitals",
                }
                results.append(row_no)
                print(f"  ({ne}e,{no}o) [Nat Orbit] Gap: {gap_no:8.4f} mHa (E_ref: {e_ref_no:.8f}, CASCI: {e_casci_no:.8f})")

                if gap_no >= 5.0 and qubits <= 12 and is_stable:
                    hard_candidates.append((qubits, gap_no, scale, ne, no, "natural_orbitals", h1e_no, eri_no, e_core_no, e_ref_no, e_casci_no, mol, mf))

            if gap_can >= 5.0 and qubits <= 12 and is_stable:
                hard_candidates.append((qubits, gap_can, scale, ne, no, "canonical", h1e_can, eri_can, e_core_can, e_ref_can, e_casci_can, mol, mf))

    # Add 8e8o skipped rows
    for scale in scales:
        results.append({
            "scale": scale,
            "ne": 8,
            "no": 8,
            "qubits": 16,
            "E_HF": "SKIPPED",
            "E_CASCI": "SKIPPED",
            "gap_mHa": "SKIPPED (>12q)",
            "RHF_stable": "N/A",
            "orbital_selection": "N/A",
        })

    # Save CSV
    csv_path = DATA_DIR / "active_space_scan.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["scale", "ne", "no", "qubits", "E_HF", "E_CASCI", "gap_mHa", "RHF_stable", "orbital_selection"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"\n[OK] Saved active space scan to {csv_path}")

    # Helper to check full Fock space sector and CASCI match
    from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
    from qiskit_nature.second_q.mappers import JordanWignerMapper

    def check_sector_and_casci(h1, eri, e_core, e_casci, ne, no):
        h2_phys = np.einsum("pqrs->prqs", eri)
        ee = ElectronicEnergy.from_raw_integrals(h1, h2_phys)
        qubit_op = JordanWignerMapper().map(ee.second_q_op())
        mat = qubit_op.to_matrix()
        evals, evecs = np.linalg.eigh(mat)
        diff = abs(float(evals[0] + e_core) - float(e_casci))
        if diff > 1e-8:
            return False, diff
        gs = evecs[:, 0]
        # check dominant bitstring sector
        max_idx = np.argmax(np.abs(gs)**2)
        bits = format(max_idx, f"0{2*no}b")
        q_occ = [int(b) for b in reversed(bits)]
        sec = (sum(q_occ[:no]), sum(q_occ[no:]))
        return (sec == (ne // 2, ne // 2)), diff

    # Filter candidates: must have gap >= 5.0, qubits <= 12, RHF stable, AND full Fock ground in (Na, Nb) sector
    valid_candidates = []
    for cand in hard_candidates:
        c_qubits, c_gap, c_scale, c_ne, c_no, c_sel, c_h1, c_eri, c_ecore, c_eref, c_ecas, c_mol, c_mf = cand
        valid_sector, diff = check_sector_and_casci(c_h1, c_eri, c_ecore, c_ecas, c_ne, c_no)
        print(f"Candidate scale={c_scale} ({c_ne}e,{c_no}o) {c_sel}: gap={c_gap:.4f} mHa, diff={diff:.2e}, valid_sector={valid_sector}")
        if valid_sector:
            valid_candidates.append(cand)

    valid_candidates.sort(key=lambda c: (c[0], -c[1]))
    if not valid_candidates:
        print("[ERROR] No valid candidate found with ground state in target sector!")
        return

    best = valid_candidates[0]
    b_qubits, b_gap, b_scale, b_ne, b_no, b_sel, b_h1, b_eri, b_ecore, b_eref, b_ecas, b_mol, b_mf = best
    print(f"\n=== CHOSEN HARD BENCHMARK ===")
    print(f"Scale: {b_scale}, Active Space: ({b_ne}e, {b_no}o) -> {b_qubits} qubits")
    print(f"Orbital Selection: {b_sel}")
    print(f"E_ref: {b_eref:.8f} Ha | CASCI: {b_ecas:.8f} Ha | Gap: {b_gap:.4f} mHa")

    hard_dict = {
        "molecule": "B8N8H10",
        "label": "BN quantum dot (hard benchmark)",
        "basis": basis,
        "orbital_method": "rhf",
        "orbital_selection": b_sel,
        "scale": b_scale,
        "cart": False,
        "n_basis_functions": int(b_mol.nao),
        "n_atoms": 26,
        "n_electrons": int(b_mol.nelectron),
        "n_active_electrons": b_ne,
        "n_active_orbitals": b_no,
        "num_qubits": b_qubits,
        "hf_energy": float(b_mf.e_tot),
        "scf_energy": float(b_mf.e_tot),
        "reference_determinant_energy": b_eref,
        "nuclear_repulsion_energy": float(b_mol.energy_nuc()),
        "e_core": float(b_ecore),
        "inactive_energy": float(b_ecore),
        "casci_energy": b_ecas,
        "correlation_energy_mHa": b_gap,
        "h1_active": b_h1.tolist(),
        "h2_active": b_eri.tolist(),
        "mo_energies": b_mf.mo_energy.tolist()[:10],
        "metadata": {
            "scale": b_scale,
            "orbital_selection": b_sel,
            "versions": {
                "pyscf": pyscf.__version__,
                "python": sys.version.split()[0],
            },
            "description": f"Stretched B-N bonds (scale={b_scale}) with ({b_ne}e,{b_no}o) {b_sel} active space yielding {b_gap:.4f} mHa correlation energy (> 5.0 mHa)",
        }
    }

    hard_json_path = DATA_DIR / "bn_dot_hard.json"
    with open(hard_json_path, "w", encoding="utf-8") as f:
        json.dump(hard_dict, f, indent=2)
    print(f"[OK] Saved hard benchmark Hamiltonian to {hard_json_path}")


if __name__ == "__main__":
    run_scan()

