"""
Aggregator and automated test runner for BN Quantum Dot VQE Verification Suite.
Runs all verification checks and outputs a structured summary table for results.xlsx.
"""
import time
import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from pathlib import Path

from src.molecule import load_or_build_bn_dot_hamiltonian, GEOMETRY_STR
from src.ansatze import get_ansatz_dict
from src.optimizers import compute_gradient_batched, run_vqe_single
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_nature.second_q.mappers import JordanWignerMapper

def run_all_verifications() -> pd.DataFrame:
    """
    Executes all verification checks programmatically and returns a DataFrame
    with columns: [Check_ID, Category, Description, Reference_Value, Computed_Value, Tolerance, Status].
    """
    records = []
    
    # -------------------------------------------------------------
    # Check 1: STO-3G RHF & CASCI Values
    # -------------------------------------------------------------
    try:
        H_sto, E_sto_casci, meta_sto = load_or_build_bn_dot_hamiltonian("sto-3g")
        rhf_sto = meta_sto["hf_energy"]
        pass_rhf = np.isclose(rhf_sto, -631.74167448, atol=1e-6)
        pass_casci = np.isclose(E_sto_casci, -631.74178346, atol=1e-6)
        records.append({
            "Check_ID": "VERIF-01A",
            "Category": "PySCF Reference",
            "Description": "STO-3G RHF Ground State Energy",
            "Reference_Value": "-631.74167448 Ha",
            "Computed_Value": f"{rhf_sto:.8f} Ha",
            "Tolerance": "1e-6 Ha",
            "Status": "PASS" if pass_rhf else "FAIL"
        })
        records.append({
            "Check_ID": "VERIF-01B",
            "Category": "PySCF Reference",
            "Description": "STO-3G CASCI (2e,2o) Ground Energy",
            "Reference_Value": "-631.74178346 Ha",
            "Computed_Value": f"{E_sto_casci:.8f} Ha",
            "Tolerance": "1e-6 Ha",
            "Status": "PASS" if pass_casci else "FAIL"
        })
    except Exception as e:
        records.append({
            "Check_ID": "VERIF-01",
            "Category": "PySCF Reference",
            "Description": "STO-3G Energies",
            "Reference_Value": "-631.74167448 Ha",
            "Computed_Value": f"Error: {e}",
            "Tolerance": "1e-6 Ha",
            "Status": "FAIL"
        })

    # -------------------------------------------------------------
    # Check 2: 6-31G(d,p) RHF & CASCI Values
    # -------------------------------------------------------------
    try:
        H_631, E_631_casci, meta_631 = load_or_build_bn_dot_hamiltonian("6-31g(d,p)")
        rhf_631 = meta_631["hf_energy"]
        pass_rhf_631 = np.isclose(rhf_631, -639.72424230, atol=1e-6)
        pass_casci_631 = np.isclose(E_631_casci, -639.72428323, atol=1e-6)
        records.append({
            "Check_ID": "VERIF-02A",
            "Category": "PySCF Reference",
            "Description": "6-31G(d,p) RHF Ground State Energy",
            "Reference_Value": "-639.72424230 Ha",
            "Computed_Value": f"{rhf_631:.8f} Ha",
            "Tolerance": "1e-6 Ha",
            "Status": "PASS" if pass_rhf_631 else "FAIL"
        })
        records.append({
            "Check_ID": "VERIF-02B",
            "Category": "PySCF Reference",
            "Description": "6-31G(d,p) CASCI (2e,2o) Ground Energy",
            "Reference_Value": "-639.72428323 Ha",
            "Computed_Value": f"{E_631_casci:.8f} Ha",
            "Tolerance": "1e-6 Ha",
            "Status": "PASS" if pass_casci_631 else "FAIL"
        })
    except Exception as e:
        records.append({
            "Check_ID": "VERIF-02",
            "Category": "PySCF Reference",
            "Description": "6-31G(d,p) Energies",
            "Reference_Value": "-639.72424230 Ha",
            "Computed_Value": f"Error: {e}",
            "Tolerance": "1e-6 Ha",
            "Status": "FAIL"
        })

    # -------------------------------------------------------------
    # Check 3: <HF|H|HF> equals RHF
    # -------------------------------------------------------------
    try:
        hf_qc = HartreeFock(2, (1, 1), JordanWignerMapper())
        sv_hf = Statevector(hf_qc)
        exp_hf_e = float(sv_hf.expectation_value(H_631).real)
        pass_hf_exp = np.isclose(exp_hf_e, -639.72424230, atol=1e-6)
        records.append({
            "Check_ID": "VERIF-03",
            "Category": "Hamiltonian Mapping",
            "Description": "<HF|H|HF> Statevector Expectation Value",
            "Reference_Value": "-639.72424230 Ha",
            "Computed_Value": f"{exp_hf_e:.8f} Ha",
            "Tolerance": "1e-6 Ha",
            "Status": "PASS" if pass_hf_exp else "FAIL"
        })
    except Exception as e:
        records.append({
            "Check_ID": "VERIF-03",
            "Category": "Hamiltonian Mapping",
            "Description": "<HF|H|HF> Expectation Value",
            "Reference_Value": "-639.72424230 Ha",
            "Computed_Value": f"Error: {e}",
            "Tolerance": "1e-6 Ha",
            "Status": "FAIL"
        })

    # -------------------------------------------------------------
    # Check 4: Sector Restriction N=2, Sz=0
    # -------------------------------------------------------------
    try:
        h_mat = H_631.to_matrix()
        valid_indices = []
        for idx in range(16):
            b = f"{idx:04b}"
            n_alpha = int(b[3]) + int(b[2])
            n_beta = int(b[1]) + int(b[0])
            if n_alpha == 1 and n_beta == 1:
                valid_indices.append(idx)
        sector_mat = h_mat[np.ix_(valid_indices, valid_indices)]
        evals, _ = np.linalg.eigh(sector_mat)
        sector_e = float(evals[0].real)
        pass_sector = np.isclose(sector_e, -639.72428323, atol=1e-6)
        records.append({
            "Check_ID": "VERIF-04",
            "Category": "Sector Diagonalization",
            "Description": "Exact Diagonalization in N=2, Sz=0 Subspace",
            "Reference_Value": "-639.72428323 Ha",
            "Computed_Value": f"{sector_e:.8f} Ha",
            "Tolerance": "1e-6 Ha",
            "Status": "PASS" if pass_sector else "FAIL"
        })
    except Exception as e:
        records.append({
            "Check_ID": "VERIF-04",
            "Category": "Sector Diagonalization",
            "Description": "Exact Diagonalization in Sector",
            "Reference_Value": "-639.72428323 Ha",
            "Computed_Value": f"Error: {e}",
            "Tolerance": "1e-6 Ha",
            "Status": "FAIL"
        })

    # -------------------------------------------------------------
    # Check 5: Ansatz Zero-Parameter Reproduction & Parameter Counts
    # -------------------------------------------------------------
    ansatze = get_ansatz_dict()
    param_expectations = {"DexcG": 1, "PCU2": 14, "UCCSD": 3, "k-UpCCGSD": 9}
    for name, qc in ansatze.items():
        actual_p = qc.num_parameters
        expected_p = param_expectations[name]
        bound = qc.assign_parameters(np.zeros(actual_p))
        sv = Statevector(bound)
        e_zero = float(sv.expectation_value(H_631).real)
        pass_p = (actual_p == expected_p)
        pass_zero = np.isclose(e_zero, -639.72424230, atol=1e-6)
        records.append({
            "Check_ID": f"VERIF-05_{name}",
            "Category": "Ansatz Structure",
            "Description": f"{name} Parameter Count & Zero-Param Energy",
            "Reference_Value": f"{expected_p} params, -639.72424230 Ha",
            "Computed_Value": f"{actual_p} params, {e_zero:.8f} Ha",
            "Tolerance": "0 params, 1e-6 Ha",
            "Status": "PASS" if (pass_p and pass_zero) else "FAIL"
        })

    # -------------------------------------------------------------
    # Check 6: Particle Number & Sz Conservation
    # -------------------------------------------------------------
    N_op = SparsePauliOp(["IIII", "IIIZ", "IIZI", "IZII", "ZIII"], [2.0, -0.5, -0.5, -0.5, -0.5])
    Sz_op = SparsePauliOp(["IIIZ", "IIZI", "IZII", "ZIII"], [-0.25, -0.25, 0.25, 0.25])
    for name in ["DexcG", "UCCSD", "k-UpCCGSD", "PCU2"]:
        qc = ansatze[name]
        res = run_vqe_single(qc, H_631, name, "zero", "GD", maxiter=15, lr=0.1)
        sv = Statevector(qc.assign_parameters(res["best_params"]))
        n_val = float(sv.expectation_value(N_op).real)
        sz_val = float(sv.expectation_value(Sz_op).real)
        pass_n = np.isclose(n_val, 2.0, atol=1e-4)
        pass_sz = np.isclose(sz_val, 0.0, atol=1e-4)
        records.append({
            "Check_ID": f"VERIF-06_{name}",
            "Category": "Symmetry Conservation",
            "Description": f"{name} Optimized State N & Sz Conservation",
            "Reference_Value": "N=2.0000, Sz=0.0000",
            "Computed_Value": f"N={n_val:.4f}, Sz={sz_val:.4f}",
            "Tolerance": "1e-4",
            "Status": "PASS" if (pass_n and pass_sz) else "FAIL"
        })

    # -------------------------------------------------------------
    # Check 7: Gradient Accuracy vs Independent Finite Differences
    # -------------------------------------------------------------
    try:
        from qiskit.primitives import StatevectorEstimator
        est = StatevectorEstimator()
        qc_ucc = ansatze["UCCSD"]
        pt = np.array([0.2, 0.3, 0.4])
        g_batched = compute_gradient_batched(qc_ucc, H_631, pt, est, method="finite_diff", eps=1e-5)
        g_indep = np.zeros(3)
        for i in range(3):
            pp, pm = np.copy(pt), np.copy(pt)
            pp[i] += 1e-5
            pm[i] -= 1e-5
            ep = float(est.run([(qc_ucc, H_631, [pp])]).result()[0].data.evs[0])
            em = float(est.run([(qc_ucc, H_631, [pm])]).result()[0].data.evs[0])
            g_indep[i] = (ep - em) / (2e-5)
        diff = np.max(np.abs(g_batched - g_indep))
        pass_grad = (diff < 1e-4)
        records.append({
            "Check_ID": "VERIF-07",
            "Category": "Gradient Verification",
            "Description": "Batched Finite Diff vs Independent Finite Diff (UCCSD)",
            "Reference_Value": "Max Diff = 0.0000",
            "Computed_Value": f"Max Diff = {diff:.2e}",
            "Tolerance": "1e-4",
            "Status": "PASS" if pass_grad else "FAIL"
        })
    except Exception as e:
        records.append({
            "Check_ID": "VERIF-07",
            "Category": "Gradient Verification",
            "Description": "Gradient Verification",
            "Reference_Value": "Max Diff < 1e-4",
            "Computed_Value": f"Error: {e}",
            "Tolerance": "1e-4",
            "Status": "FAIL"
        })

    # -------------------------------------------------------------
    # Check 8: Reproducibility Across Identical Seeds
    # -------------------------------------------------------------
    try:
        r1 = run_vqe_single(ansatze["UCCSD"], H_631, "UCCSD", "random", "SPSA", maxiter=10, seed=42)
        r2 = run_vqe_single(ansatze["UCCSD"], H_631, "UCCSD", "random", "SPSA", maxiter=10, seed=42)
        diff_e = abs(r1["final_energy"] - r2["final_energy"])
        pass_rep = (diff_e < 1e-10)
        records.append({
            "Check_ID": "VERIF-08",
            "Category": "Reproducibility",
            "Description": "Identical Seed VQE SPSA Run Agreement",
            "Reference_Value": "Delta E = 0.0000 Ha",
            "Computed_Value": f"Delta E = {diff_e:.2e} Ha",
            "Tolerance": "1e-10 Ha",
            "Status": "PASS" if pass_rep else "FAIL"
        })
    except Exception as e:
        records.append({
            "Check_ID": "VERIF-08",
            "Category": "Reproducibility",
            "Description": "Reproducibility Test",
            "Reference_Value": "Delta E < 1e-10 Ha",
            "Computed_Value": f"Error: {e}",
            "Tolerance": "1e-10 Ha",
            "Status": "FAIL"
        })

    # -------------------------------------------------------------
    # Check 9: Molecular Geometry Bonds
    # -------------------------------------------------------------
    lines = [l.strip().split() for l in GEOMETRY_STR.strip().split("\n") if l.strip()]
    atoms = [(l[0], np.array([float(l[1]), float(l[2]), float(l[3])])) for l in lines]
    bn_dists = [np.linalg.norm(atoms[i][1] - atoms[j][1]) for i in range(len(atoms)) for j in range(i+1, len(atoms)) if tuple(sorted([atoms[i][0], atoms[j][0]])) == ("B", "N") and np.linalg.norm(atoms[i][1] - atoms[j][1]) < 1.7]
    mean_bn = float(np.mean(bn_dists))
    pass_geom = np.isclose(mean_bn, 1.44, atol=0.05)
    records.append({
        "Check_ID": "VERIF-09",
        "Category": "Molecular Geometry",
        "Description": "B-N Mean Bond Length",
        "Reference_Value": "1.4400 A",
        "Computed_Value": f"{mean_bn:.4f} A",
        "Tolerance": "0.05 A",
        "Status": "PASS" if pass_geom else "FAIL"
    })

    df = pd.DataFrame(records)
    return df

if __name__ == "__main__":
    df_verif = run_all_verifications()
    print("\n==========================================================================")
    print("BN QUANTUM DOT VQE VERIFICATION SUITE RESULTS")
    print("==========================================================================")
    print(df_verif.to_string(index=False))
    print("==========================================================================")
    all_passed = (df_verif["Status"] == "PASS").all()
    print(f"Overall Verification Status: {'ALL PASSED [PASS]' if all_passed else 'SOME FAILED [FAIL]'}")

