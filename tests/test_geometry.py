"""
Verification Test: Molecular geometry of BN Quantum Dot (B8N8H10).
Verifies bond lengths: B-N ~1.44 A, B-H ~1.19 A, N-H ~1.01 A (within physical tolerances).
"""
import numpy as np
from src.molecule import GEOMETRY_STR

def test_molecular_geometry_bond_lengths():
    lines = [l.strip().split() for l in GEOMETRY_STR.strip().split("\n") if l.strip()]
    atoms = [(l[0], np.array([float(l[1]), float(l[2]), float(l[3])])) for l in lines]
    assert len(atoms) == 26, f"Expected 26 atoms, found {len(atoms)}"
    
    bn_dists, bh_dists, nh_dists = [], [], []
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            sym1, pos1 = atoms[i]
            sym2, pos2 = atoms[j]
            dist = np.linalg.norm(pos1 - pos2)
            pair = tuple(sorted([sym1, sym2]))
            if pair == ("B", "N") and dist < 1.7:
                bn_dists.append(dist)
            elif pair == ("B", "H") and dist < 1.4:
                bh_dists.append(dist)
            elif pair == ("H", "N") and dist < 1.3:
                nh_dists.append(dist)
                
    assert len(bn_dists) > 0, "No B-N bonds found"
    assert len(bh_dists) > 0, "No B-H bonds found"
    assert len(nh_dists) > 0, "No N-H bonds found"
    
    mean_bn = float(np.mean(bn_dists))
    mean_bh = float(np.mean(bh_dists))
    mean_nh = float(np.mean(nh_dists))
    
    # B-N ~1.44 A (within 0.05 A)
    assert np.isclose(mean_bn, 1.44, atol=0.05), f"B-N mean bond length {mean_bn:.4f} A deviates from 1.44 A"
    # B-H ~1.19 A (within 0.05 A)
    assert np.isclose(mean_bh, 1.19, atol=0.05), f"B-H mean bond length {mean_bh:.4f} A deviates from 1.19 A"
    # N-H ~1.01-1.05 A (within 0.08 A)
    assert 0.95 <= mean_nh <= 1.15, f"N-H mean bond length {mean_nh:.4f} A outside expected range"

