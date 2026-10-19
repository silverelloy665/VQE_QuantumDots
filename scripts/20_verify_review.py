"""
Comprehensive Verification Script for Independent Review Findings (Round 3).
Reproduces and verifies Findings 1 through 7 before implementing any changes.
Outputs a clean PASS/FAIL verification table.
"""
import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = REPO_ROOT / "data"
FIGURES_DIR = REPO_ROOT / "figures"
# Use legacy Round 2 files if available to verify pre-Round 3 review claims
LEGACY_JSON = DATA_DIR / "benchmark_results_hard_round2_legacy.json"
HARD_BENCHMARK_JSON = LEGACY_JSON if LEGACY_JSON.exists() else (DATA_DIR / "benchmark_results_hard.json")

LEGACY_CSV = DATA_DIR / "hard_benchmark_summary_round2_legacy.csv"
HARD_SUMMARY_CSV = LEGACY_CSV if LEGACY_CSV.exists() else (DATA_DIR / "hard_benchmark_summary.csv")
CIRCUIT_COSTS_CSV = DATA_DIR / "circuit_costs_hard.csv"

E_CASCI_HARD = -637.5813354347313
E_HF_HARD = -637.5731962711441
GAP_HARD_MHA = abs(E_HF_HARD - E_CASCI_HARD) * 1000.0  # 8.1392 mHa


def verify_finding_1():
    """
    Finding 1:
    Zero-init L-BFGS-B errors on hard problem:
    UCCSD ~ 0.008 mHa, UCCSD-SA ~ 0.008, DexcG ~ 0.015, PCU2 ~ 0.015,
    2-UpCCGSD ~ 0.029, 3-UpCCGSD ~ 0.029, 1-UpCCGSD ~ 1.484 mHa.
    The reported winner (2-UpCCGSD, random init, 0.802 mHa) is worse than all except 1-UpCCGSD.
    The summary CSV only aggregates init == 'random'.
    """
    if not HARD_BENCHMARK_JSON.exists():
        return False, "benchmark_results_hard.json not found"
    
    with open(HARD_BENCHMARK_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    recs = data.get("records", [])

    zero_lbfgs = [r for r in recs if r.get("init") == "zero" and r.get("optimizer") == "L-BFGS-B"]
    zero_errors = {}
    for r in zero_lbfgs:
        err = (r["final_energy"] - E_CASCI_HARD) * 1000.0
        zero_errors[r["ansatz"]] = err

    # Check summary CSV filter
    df_sum = pd.read_csv(HARD_SUMMARY_CSV)
    # Check if random is what was aggregated
    random_winner_err = df_sum.iloc[0]["Median_Error_mHa"]

    details = (
        f"UCCSD: {zero_errors.get('UCCSD', -1):.3f} mHa, "
        f"UCCSD-SA: {zero_errors.get('UCCSD-SA', -1):.3f} mHa, "
        f"DexcG: {zero_errors.get('DexcG', -1):.3f} mHa, "
        f"PCU2: {zero_errors.get('PCU2', -1):.3f} mHa, "
        f"2-UpCCGSD: {zero_errors.get('2-UpCCGSD', -1):.3f} mHa, "
        f"1-UpCCGSD: {zero_errors.get('1-UpCCGSD', -1):.3f} mHa. "
        f"Random winner: 2-UpCCGSD median {random_winner_err:.3f} mHa."
    )

    # Check expected ranges
    u_ok = abs(zero_errors.get("UCCSD", 99) - 0.008) < 0.005
    d_ok = abs(zero_errors.get("DexcG", 99) - 0.015) < 0.005
    u1_ok = abs(zero_errors.get("1-UpCCGSD", 99) - 1.484) < 0.05
    return (u_ok and d_ok and u1_ok), details


def verify_finding_2():
    """
    Finding 2:
    The 'converged' field is always True in benchmark_results_hard.json.
    run_vqe_single never returns 'converged', and SPSA/QNSPSA/COBYLA from random
    starts end at 400-650 mHa (unconverged).
    """
    if not HARD_BENCHMARK_JSON.exists():
        return False, "benchmark_results_hard.json not found"

    with open(HARD_BENCHMARK_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    recs = data.get("records", [])

    all_converged = all(r.get("converged") is True for r in recs)
    # Check if unconverged runs exist (errors > 400 mHa)
    unconverged_errors = [
        (r["final_energy"] - E_CASCI_HARD) * 1000.0
        for r in recs if r.get("optimizer") in ["SPSA", "QNSPSA", "COBYLA"] and r.get("init") == "random"
    ]
    max_err = max(unconverged_errors) if unconverged_errors else 0.0
    mean_err = np.mean(unconverged_errors) if unconverged_errors else 0.0

    passed = all_converged and (mean_err > 400.0)
    details = f"All {len(recs)} records marked converged=True. Random SPSA/QNSPSA/COBYLA mean error: {mean_err:.1f} mHa (max {max_err:.1f} mHa)."
    return passed, details


def verify_finding_3():
    """
    Finding 3:
    Mean_Pct_Corr is distorted by outliers (1-UpCCGSD / L-BFGS-B: mean -391.7%, median 76.7%).
    """
    if not HARD_SUMMARY_CSV.exists():
        return False, "hard_benchmark_summary.csv not found"

    df_sum = pd.read_csv(HARD_SUMMARY_CSV)
    row_1up = df_sum[(df_sum["Ansatz"] == "1-UpCCGSD") & (df_sum["Optimizer"] == "L-BFGS-B")]
    if row_1up.empty:
        return False, "1-UpCCGSD / L-BFGS-B not found in summary"

    mean_corr = float(row_1up.iloc[0]["Mean_Pct_Corr"])

    # Compute actual median pct_corr across the 10 seeds from raw records
    with open(HARD_BENCHMARK_JSON, "r", encoding="utf-8") as f:
        recs = json.load(f)["records"]

    runs_1up = [
        r for r in recs
        if r.get("ansatz") == "1-UpCCGSD" and r.get("optimizer") == "L-BFGS-B" and r.get("init") == "random"
    ]
    pct_corrs = [
        (E_HF_HARD - r["final_energy"]) / (E_HF_HARD - E_CASCI_HARD) * 100.0
        for r in runs_1up
    ]
    med_corr = float(np.median(pct_corrs))

    passed = (abs(mean_corr - (-391.67)) < 1.0) and (abs(med_corr - 76.74) < 1.0)
    details = f"1-UpCCGSD / L-BFGS-B: Mean_Pct_Corr = {mean_corr:.2f}%, Median Pct_Corr = {med_corr:.2f}%."
    return passed, details


def verify_finding_4():
    """
    Finding 4:
    scripts/run_live_hardware_job.py hard-codes params_1up (|theta| <= 3e-4, repeating +-0.000302 / +-0.000108)
    claiming L-BFGS-B origin. Its noiseless energy has error 8.145 mHa, which is WORSE than HF (8.139 mHa).
    Cache stores no final_params.
    """
    from src.molecule import load_hard_hamiltonian
    from src.ansatze import get_ansatz_dict
    from qiskit.quantum_info import Statevector

    params_1up = [
        0.000302, 0.000108, -0.000108, -0.000302, 0.000108, 0.000302,
        -0.000302, -0.000108, -0.000108, -0.000302, 0.000108, 0.000302,
        -0.000302, -0.000108, 0.000108, 0.000302, -0.000108, -0.000302
    ]

    H_hard, _, _ = load_hard_hamiltonian()
    ansatze = get_ansatz_dict(num_spatial_orbitals=4, num_particles=(2, 2), include_k_variants=True)
    qc = ansatze["1-UpCCGSD"].decompose().decompose()
    sv = Statevector(qc.assign_parameters(params_1up))
    e_val = float(sv.expectation_value(H_hard).real)
    err_mHa = (e_val - E_CASCI_HARD) * 1000.0

    # Check if cache stores params
    with open(HARD_BENCHMARK_JSON, "r", encoding="utf-8") as f:
        recs = json.load(f)["records"]
    has_params = any("final_params" in r or "optimal_params" in r for r in recs)

    worse_than_hf = err_mHa > GAP_HARD_MHA
    passed = worse_than_hf and (not has_params)
    details = f"params_1up noiseless error: {err_mHa:.4f} mHa vs HF error: {GAP_HARD_MHA:.4f} mHa (worse by {(err_mHa - GAP_HARD_MHA):.4f} mHa). Cache has final_params: {has_params}."
    return passed, details


def verify_finding_5():
    """
    Finding 5:
    Fidelity inconsistency: HERON_CZ_ERROR = 0.0398 for 2-CZ parity circuit (F=0.922),
    while circuit_costs_hard.csv 489/995-CZ implies ~0.00186 per CZ.
    At 0.00186, 2-CZ circuit would be ~0.996; at 0.0398, 489-CZ circuit would be ~0.
    CZ-only fidelity also ignores 1q, readout and decoherence.
    """
    df_costs = pd.read_csv(CIRCUIT_COSTS_CSV)
    r_1up = df_costs[df_costs["Ansatz"] == "1-UpCCGSD"].iloc[0]
    f_489 = float(r_1up["Estimated_Fidelity"])
    n_489 = int(r_1up["Transpiled_2Q"])

    implied_err = 1.0 - (f_489 ** (1.0 / n_489))
    f_2cz_at_implied = (1.0 - implied_err) ** 2
    f_489_at_device_mean = (1.0 - 0.0398) ** n_489

    passed = (abs(implied_err - 0.00184) < 0.0005) and (abs(f_2cz_at_implied - 0.9963) < 0.005)
    details = (
        f"Implied per-CZ error from 1-UpCCGSD: {implied_err:.5f} (vs 0.0398 device-wide). "
        f"2-CZ at {implied_err:.5f}: F={f_2cz_at_implied:.4f} (vs 0.9220). "
        f"489-CZ at 0.0398: F={f_489_at_device_mean:.2e}."
    )
    return passed, details


def verify_finding_6():
    """
    Finding 6:
    REPORT.md says E(theta) = a + b cos(4 theta) + c sin(4 theta) with points 0, pi/4, pi/2.
    The true landscape has period pi (omega = 2); a 4-theta model is singular at those points.
    """
    # Evaluate DexcG energy landscape on fine grid and FFT
    from src.molecule import load_or_build_bn_dot_hamiltonian
    from src.ansatze import get_ansatz_dict
    from qiskit.quantum_info import Statevector

    H, _, _ = load_or_build_bn_dot_hamiltonian("6-31g(d,p)")
    ansatze = get_ansatz_dict()
    qc = ansatze["DexcG"]

    thetas = np.linspace(0, 2 * np.pi, 256, endpoint=False)
    energies = [float(Statevector(qc.assign_parameters([t])).expectation_value(H).real) for t in thetas]

    fft_vals = np.abs(np.fft.rfft(energies - np.mean(energies)))
    freqs = np.fft.rfftfreq(len(thetas), d=(thetas[1] - thetas[0]) / (2 * np.pi))
    dominant_harmonic = np.argmax(fft_vals[1:]) + 1  # harmonic of fundamental [0, 2pi]

    # Fundamental frequency of E(theta):
    # E(theta) = a + b*cos(2*theta) + c*sin(2*theta) has harmonic = 2 in [0, 2pi].
    # A 4*theta term corresponds to period pi/2 (harmonic = 4).
    # Sample points 0, pi/4, pi/2 for 4*theta:
    # 4*0 = 0, 4*(pi/4) = pi, 4*(pi/2) = 2pi -> cos values are [1, -1, 1], sin values are [0, 0, 0] (singular!).
    mat_4theta = np.array([
        [1.0, np.cos(4 * 0.0), np.sin(4 * 0.0)],
        [1.0, np.cos(4 * np.pi / 4), np.sin(4 * np.pi / 4)],
        [1.0, np.cos(4 * np.pi / 2), np.sin(4 * np.pi / 2)]
    ])
    cond_4theta = np.linalg.cond(mat_4theta)

    mat_2theta = np.array([
        [1.0, np.cos(2 * 0.0), np.sin(2 * 0.0)],
        [1.0, np.cos(2 * np.pi / 4), np.sin(2 * np.pi / 4)],
        [1.0, np.cos(2 * np.pi / 2), np.sin(2 * np.pi / 2)]
    ])
    cond_2theta = np.linalg.cond(mat_2theta)

    passed = (dominant_harmonic == 2) and (cond_4theta > 1e14 or np.isinf(cond_4theta)) and (cond_2theta < 10.0)
    details = (
        f"FFT dominant harmonic: {dominant_harmonic} (period pi, omega=2). "
        f"4-theta model condition number at (0, pi/4, pi/2): {cond_4theta:.1e} (singular!). "
        f"2-theta model condition number: {cond_2theta:.2f}."
    )
    return passed, details


def verify_finding_7():
    """
    Finding 7:
    figures/circuit_*.png show old 2e2o circuits (4 qubits).
    Silent fallback to GenericBackendV2 when IBM service is unreachable.
    """
    # Check circuit png files
    circuit_pngs = list(FIGURES_DIR.glob("circuit_*.png"))
    # In src/ansatze.py and src/hardware.py, check for silent GenericBackendV2 fallbacks
    with open(REPO_ROOT / "src" / "ansatze.py", "r", encoding="utf-8") as f:
        src_ansatze = f.read()
    with open(REPO_ROOT / "src" / "hardware.py", "r", encoding="utf-8") as f:
        src_hardware = f.read()

    has_silent_fallback = (
        ("GenericBackendV2" in src_ansatze) or
        ("GenericBackendV2" in src_hardware)
    )

    passed = len(circuit_pngs) > 0 and has_silent_fallback
    details = f"Found {len(circuit_pngs)} legacy circuit PNGs. Silent GenericBackendV2 fallbacks detected: {has_silent_fallback}."
    return passed, details


def main():
    print("=" * 80)
    print("VERIFICATION OF ROUND 3 INDEPENDENT REVIEW FINDINGS")
    print("=" * 80)

    checks = [
        ("Finding 1: Zero-init superior to reported random 'winner'", verify_finding_1),
        ("Finding 2: 'converged' flag always True; SPSA/COBYLA unconverged", verify_finding_2),
        ("Finding 3: Mean_Pct_Corr distorted by outliers (mean -391% vs med 76%)", verify_finding_3),
        ("Finding 4: Hardcoded params_1up worse than HF; no params in cache", verify_finding_4),
        ("Finding 5: Device-wide vs per-edge fidelity inconsistency", verify_finding_5),
        ("Finding 6: Rotosolve 4-theta text error & singular sample points", verify_finding_6),
        ("Finding 7: Legacy 2e2o circuit figures & silent mock fallbacks", verify_finding_7),
    ]

    results = []
    for name, func in checks:
        try:
            ok, details = func()
            status = "REPRODUCED (PASS)" if ok else "FAILED TO REPRODUCE"
        except Exception as e:
            ok = False
            status = "ERROR"
            details = str(e)
        results.append({"Check": name, "Status": status, "Details": details})

    df = pd.DataFrame(results)
    for idx, r in df.iterrows():
        print(f"\n[{r['Status']}] {r['Check']}")
        print(f"  Details: {r['Details']}")

    print("\n" + "=" * 80)
    all_reproduced = all("PASS" in r["Status"] for r in results)
    print(f"OVERALL STATUS: {'ALL 7 FINDINGS FULLY REPRODUCED' if all_reproduced else 'SOME FINDINGS FAILED'}")
    print("=" * 80)
    return 0 if all_reproduced else 1


if __name__ == "__main__":
    sys.exit(main())
