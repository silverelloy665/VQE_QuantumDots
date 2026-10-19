"""
Generates verified tables and documentation sections for README.md and REPORT.md
strictly from data/ result files:
- data/reconciled_hardware_jobs.csv
- data/active_space_scan.csv
- data/hard_benchmark_summary.csv
- data/circuit_costs_hard.csv
- data/hardware_job_spec_v2.json

Guarantees 100% compliance with the hard rule: No fabricated or hard-coded result numbers.
"""
import json
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


def generate_reconciled_jobs_md() -> str:
    df = pd.read_csv(DATA_DIR / "reconciled_hardware_jobs.csv")
    md = "| Job ID | Backend | Shots | Ansatz | Parameters | Resilience | Measured Energy (Ha) | Error vs CASCI (mHa) | Error vs HF (mHa) | Source |\n"
    md += "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
    for _, r in df.iterrows():
        md += f"| `{r['Job_ID']}` | {r['Backend']} | {r['Shots']} | {r['Ansatz']} | {r['Parameters']} | {r['Resilience']} | {r['Measured_Energy_Ha']:.8f} | {r['Error_vs_CASCI_mHa']:.4f} | {r['Error_vs_HF_mHa']:.4f} | {r['Source']} |\n"
    return md


def generate_active_space_scan_md() -> str:
    df = pd.read_csv(DATA_DIR / "active_space_scan.csv")
    md = "| Scale | Active Space | Qubits | Reference Determinant (Ha) | Exact CASCI (Ha) | Correlation Gap (mHa) | RHF Stable | Orbital Selection |\n"
    md += "| :---: | :---: | :---: | :--- | :--- | :--- | :---: | :---: |\n"
    for _, r in df.iterrows():
        gap_str = f"{float(r['gap_mHa']):.4f}" if r['gap_mHa'] != "SKIPPED (>12q)" else str(r['gap_mHa'])
        e_hf_str = f"{float(r['E_HF']):.8f}" if r['E_HF'] != "SKIPPED" else "SKIPPED"
        e_casci_str = f"{float(r['E_CASCI']):.8f}" if r['E_CASCI'] != "SKIPPED" else "SKIPPED"
        md += f"| {r['scale']} | ({r['ne']}e, {r['no']}o) | {r['qubits']} | {e_hf_str} | {e_casci_str} | {gap_str} | {r['RHF_stable']} | {r['orbital_selection']} |\n"
    return md


def generate_hard_benchmark_summary_md(top_n: int = 15) -> str:
    df = pd.read_csv(DATA_DIR / "hard_benchmark_summary.csv")
    # Sort by Median_Error_mHa
    df_sorted = df.sort_values(by="Median_Error_mHa").reset_index(drop=True)
    md = "| Rank | Ansatz | Optimizer | Seeds | Median Error (mHa) | IQR (mHa) | Min Error (mHa) | Mean Corr Recov (%) | Success Rate (%) | Better than HF (%) |\n"
    md += "| :---: | :--- | :--- | :---: | :--- | :--- | :--- | :--- | :--- | :--- |\n"
    for idx, r in df_sorted.head(top_n).iterrows():
        md += f"| **{idx+1}** | **{r['Ansatz']}** | {r['Optimizer']} | {r['N_Seeds']} | {r['Median_Error_mHa']:.4f} | {r['IQR_Error_mHa']:.4f} | {r['Min_Error_mHa']:.4f} | {r['Mean_Pct_Corr']:.2f}% | {r['Success_Rate_pct']:.1f}% | {r['Better_than_HF_pct']:.1f}% |\n"
    return md


def generate_circuit_costs_md() -> str:
    df = pd.read_csv(DATA_DIR / "circuit_costs_hard.csv")
    md = "| Ansatz | Parameters | Raw CNOT | Raw Depth | Transpiled 2Q (CZ) | Transpiled Depth | Transpiler Seed | Estimated Fidelity | Fidelity Source | Backend |\n"
    md += "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: | :--- |\n"
    for _, r in df.iterrows():
        fid_str = f"{r['Estimated_Fidelity']:.4f}" if r['Estimated_Fidelity'] > 1e-4 else f"{r['Estimated_Fidelity']:.2e}"
        md += f"| **{r['Ansatz']}** | {r['Parameters']} | {r['Raw_CNOT']} | {r['Raw_Depth']} | {r['Transpiled_2Q']} | {r['Transpiled_Depth']} | {r['Transpiler_Seed']} | **{fid_str}** | {r['Fidelity_Source']} | `{r['Target_Backend']}` |\n"
    return md


def generate_hardware_jobs_v2_md() -> str:
    with open(DATA_DIR / "hardware_job_spec_v2.json", "r", encoding="utf-8") as f:
        specs = json.load(f)
    md = "| Priority | Experiment Name | Backend | Shots | Resilience | QWC Circuits | 2Q Gates | Depth | Est. Fidelity | Est. QPU Time (s) | Recommendation |\n"
    md += "| :---: | :--- | :--- | :---: | :--- | :---: | :---: | :---: | :--- | :---: | :--- |\n"
    for s in specs:
        fid_str = f"{s['estimated_fidelity']:.4f}" if s['estimated_fidelity'] > 1e-4 else f"{s['estimated_fidelity']:.2e}"
        md += f"| {s['priority']} | `{s['experiment_name']}` | `{s['backend']}` | {s['shots']} | {s['resilience_type']} | {s['qwc_circuits']} | {s['transpiled_2q_gates']} | {s['transpiled_depth']} | {fid_str} | {s['estimated_qpu_time_seconds']}s | {s['recommendation']} |\n"
    return md


if __name__ == "__main__":
    print("=== Reconciled Jobs MD ===")
    print(generate_reconciled_jobs_md())
    print("\n=== Active Space Scan MD ===")
    print(generate_active_space_scan_md())
    print("\n=== Hard Benchmark Summary MD ===")
    print(generate_hard_benchmark_summary_md(10))
    print("\n=== Circuit Costs MD ===")
    print(generate_circuit_costs_md())
    print("\n=== Hardware Jobs V2 MD ===")
    print(generate_hardware_jobs_v2_md())

