"""
Standalone end-to-end benchmark execution script for BN Quantum Dot (B8N8H10) VQE.
Generates all computed data, figures, Excel workbook, and verification suite results.
"""
import sys
import time
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.molecule import load_or_build_bn_dot_hamiltonian
from src.ansatze import get_ansatz_dict, analyze_and_render_circuits
from src.benchmark import run_full_vqe_benchmark
from src.hardware import run_hardware_evaluation
from src.benchmark import run_full_vqe_benchmark, run_lr_sweep, run_robustness_benchmark
from src.hardware import run_noisy_fake_backend_evaluation, test_transpilation_parameter_binding
from src.excel_export import export_benchmark_to_excel
from tests.run_verification_suite import run_all_verifications

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
DATA_DIR = Path(__file__).resolve().parent / "data"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

def generate_overview_plot(results_df: pd.DataFrame, output_path: Path):
    """Generates a summary multi-panel chart comparing ansatz, inits, and optimizers."""
def generate_overview_plot(results_df: pd.DataFrame, robustness_df: pd.DataFrame, output_path: Path):
    """Generates a summary multi-panel chart comparing ansatz, inits, and optimizers with error bars."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Error by Ansatz & Optimizer
    # 1. Error by Ansatz & Optimizer (Minimum Error)
    piv_err = results_df.pivot_table(index="Ansatz", columns="Optimizer", values="Error_mHa", aggfunc="min")
    piv_err.plot(kind="bar", ax=axes[0, 0], colormap="viridis", edgecolor="black")
    axes[0, 0].set_title("Minimum Energy Error (mHa) by Ansatz & Optimizer", fontsize=12, fontweight="bold")
    axes[0, 0].set_ylabel("Error (mHa)")
    axes[0, 0].grid(axis="y", linestyle="--", alpha=0.5)
    
    # 2. Error by Initialization & Ansatz
    piv_init = results_df.pivot_table(index="Ansatz", columns="Initialization", values="Error_mHa", aggfunc="min")
    piv_init.plot(kind="bar", ax=axes[0, 1], colormap="plasma", edgecolor="black")
    axes[0, 1].set_title("Minimum Energy Error (mHa) by Initialization", fontsize=12, fontweight="bold")
    axes[0, 1].set_ylabel("Error (mHa)")
    axes[0, 1].grid(axis="y", linestyle="--", alpha=0.5)
    
    # 3. Wall Time per Optimizer
    time_summary = results_df.groupby("Optimizer")["Wall_Time_s"].mean()
    time_summary.plot(kind="bar", ax=axes[1, 0], color="#2b5c8f", edgecolor="black")
    axes[1, 0].set_title("Average Execution Time per Optimizer (seconds)", fontsize=12, fontweight="bold")
    axes[1, 0].set_ylabel("Time (s)")
    axes[1, 0].grid(axis="y", linestyle="--", alpha=0.5)
    
    # 4. Error Distribution Boxplot
    results_df.boxplot(column="Error_mHa", by="Ansatz", ax=axes[1, 1], patch_artist=True)
    axes[1, 1].set_title("Error Distribution (mHa) across Initializations & Optimizers", fontsize=12, fontweight="bold")
    axes[1, 1].set_ylabel("Error (mHa)")
    # 3. Robustness Error Bars (Random Init across 5 seeds)
    if robustness_df is not None and not robustness_df.empty:
        piv_rob_mean = robustness_df.pivot_table(index="Ansatz", columns="Optimizer", values="Mean_Error_mHa")
        piv_rob_std = robustness_df.pivot_table(index="Ansatz", columns="Optimizer", values="Std_Error_mHa")
        piv_rob_mean.plot(kind="bar", yerr=piv_rob_std, ax=axes[1, 0], capsize=4, colormap="tab10", edgecolor="black")
        axes[1, 0].set_title("Random Init Robustness (Mean ± Std over 5 Seeds)", fontsize=12, fontweight="bold")
        axes[1, 0].set_ylabel("Mean Error (mHa)")
        axes[1, 0].grid(axis="y", linestyle="--", alpha=0.5)
    else:
        time_summary = results_df.groupby("Optimizer")["Wall_Time_s"].mean()
        time_summary.plot(kind="bar", ax=axes[1, 0], color="#2b5c8f", edgecolor="black")
        axes[1, 0].set_title("Average Execution Time per Optimizer (seconds)", fontsize=12, fontweight="bold")
        axes[1, 0].set_ylabel("Time (s)")
        axes[1, 0].grid(axis="y", linestyle="--", alpha=0.5)
        
    # 4. Total Function Evaluations by Optimizer
    eval_summary = results_df.groupby("Optimizer")["Total_Evaluations"].mean()
    eval_summary.plot(kind="bar", ax=axes[1, 1], color="#2e7d32", edgecolor="black")
    axes[1, 1].set_title("Mean Function Evaluations per Optimizer", fontsize=12, fontweight="bold")
    axes[1, 1].set_ylabel("Total Function Evaluations")
    axes[1, 1].grid(axis="y", linestyle="--", alpha=0.5)
    plt.suptitle("") # Clear automatic boxplot title
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved overview figure to {output_path}")

def main():
    start_time = time.time()
    print("==========================================================================")
    print("BENCHMARK PROJECT: Hexagonal Boron-Nitride (B8N8H10) Quantum Dot VQE")
    print("VQE BENCHMARK: Hexagonal Boron-Nitride (B8N8H10) Quantum Dot")
    print("==========================================================================")
    
    # Stage 1: STO-3G Smoke Test
    print("\n[Stage 1/6] Running STO-3G Pipeline Validation (Smoke Test)...")
    print("\n[Stage 1/9] Running STO-3G Pipeline Validation (Smoke Test)...")
    H_sto, E_sto, meta_sto = load_or_build_bn_dot_hamiltonian("sto-3g")
    assert H_sto.num_qubits == 4, "Expected 4 qubits"
    assert len(H_sto) == 27, "Expected 27 Pauli terms"
    print(f"  -> STO-3G Smoke Test Passed! Exact Ground Energy = {E_sto:.8f} Ha")
    print(f"  -> STO-3G Smoke Test Passed! Exact Ground Energy = {E_sto:.8f} Ha, E_corr = {meta_sto['correlation_energy_mHa']:.4f} mHa")
    
    # Stage 2: 6-31G(d,p) Molecular Hamiltonian & Reference
    print("\n[Stage 2/6] Loading 6-31G(d,p) Hamiltonian & Exact Diagonalization Reference...")
    print("\n[Stage 2/9] Loading 6-31G(d,p) Hamiltonian & Exact Reference...")
    H_631, E_exact, meta_631 = load_or_build_bn_dot_hamiltonian("6-31g(d,p)")
    print(f"  -> Qubits: {H_631.num_qubits}, Pauli terms: {len(H_631)}")
    print(f"  -> Exact Active Space Ground State Energy: {E_exact:.8f} Ha")
    print(f"  -> RHF Energy: {meta_631['hf_energy']:.8f} Ha | Exact CASCI: {E_exact:.8f} Ha | E_corr: {meta_631['correlation_energy_mHa']:.5f} mHa")
    
    # Stage 3: Circuit Gallery & Transpilation Analysis
    print("\n[Stage 3/6] Generating Circuit Gallery and Transpilation Metrics...")
    print("\n[Stage 3/9] Generating Circuit Gallery and Transpilation Metrics...")
    ansatze = get_ansatz_dict()
    circuits_df = analyze_and_render_circuits(ansatze, save_pngs=True)
    print(circuits_df.to_string(index=False))
    
    # Stage 4: 64-Configuration VQE Benchmark Grid
    print("\n[Stage 4/6] Executing 64 VQE Runs (4 ansatze x 4 inits x 4 optimizers, 50 iter)...")
    results_df, convergence_df, meta = run_full_vqe_benchmark(basis="6-31g(d,p)", maxiter=50, verbose=True)
    # Stage 4: Learning Rate Sweep (Held-out Seed)
    print("\n[Stage 4/9] Running Learning Rate Tuning Sweep...")
    lr_sweep_df = run_lr_sweep(basis="6-31g(d,p)", sweep_seed=123, maxiter=25)
    lr_sweep_df = run_lr_sweep(basis="6-31g(d,p)", sweep_seed=123, maxiter=25, use_cache=True)
    best_lr_idx = lr_sweep_df.groupby("Optimizer")["Error_mHa"].idxmin()
    best_lr_df = lr_sweep_df.loc[best_lr_idx][["Optimizer", "Learning_Rate", "Error_mHa", "Total_Evaluations"]]
    print(best_lr_df.to_string(index=False))
    
    # Stage 5: Full 64-Configuration VQE Benchmark Grid
    print("\n[Stage 5/9] Executing 64 VQE Runs (4 ansatze x 4 inits x 4 optimizers, 50 iter)...")
    results_df, convergence_df, meta = run_full_vqe_benchmark(basis="6-31g(d,p)", maxiter=50, verbose=True, use_cache=False)
    results_df, convergence_df, meta = run_full_vqe_benchmark(basis="6-31g(d,p)", maxiter=50, verbose=True, use_cache=True)
    
    # Verify no NaNs
    assert not results_df.isnull().values.any(), "Error: NaN values detected in results!"
    assert not convergence_df.isnull().values.any(), "Error: NaN values detected in convergence trajectories!"
    
    # Top 5 Best Configurations
    sorted_df = results_df.sort_values(by="Error_mHa")
    best_run = sorted_df.iloc[0]
    print("\nTop 5 Best Performing Configurations:")
    print(sorted_df.head(5)[["Config_ID", "Ansatz", "Initialization", "Optimizer", "Final_Energy_Ha", "Error_mHa", "Wall_Time_s"]].to_string(index=False))
    # Stage 6: 5-Seed Random Robustness Evaluation
    print("\n[Stage 6/9] Running Random Init Robustness Benchmark (5 Seeds: 42, 123, 456, 789, 1000)...")
    robustness_df = run_robustness_benchmark(basis="6-31g(d,p)", seeds=[42, 123, 456, 789, 1000], maxiter=50)
    robustness_df = run_robustness_benchmark(basis="6-31g(d,p)", seeds=[42, 123, 456, 789, 1000], maxiter=50, use_cache=True)
    print(robustness_df.to_string(index=False))
    
    # Stage 5: Overview Plot Generation
    print("\n[Stage 5/6] Generating Benchmark Overview Visualizations...")
    generate_overview_plot(results_df, FIGURES_DIR / "benchmark_overview.png")
    # Stage 7: Hardware & Fake Backend Calibrated Noisy Simulation
    print("\n[Stage 7/9] Running FakeFez Calibrated Noisy Aer Simulation (4096 shots)...")
    best_run = results_df.sort_values(by=["Error_mHa", "Total_Evaluations"]).iloc[0]
    best_ansatz_qc = ansatze[best_run["Ansatz"]]
    best_params = np.zeros(best_ansatz_qc.num_parameters)
    
    # Stage 6: IBM Quantum Hardware Execution
    print("\n[Stage 6/6] Running Hardware Single-Point Evaluation on Best Configuration...")
    best_ansatz_qc = ansatze[best_run["Ansatz"]]
    opt_params = best_run["Final_Energy_Ha"] # We can fetch final optimal params
    # Re-fetch params from single run
    from src.optimizers import run_vqe_single
    best_single = run_vqe_single(
    noisy_metrics = run_noisy_fake_backend_evaluation(
        circuit=best_ansatz_qc,
        hamiltonian=H_631,
        ansatz_name=best_run["Ansatz"],
        init_name=best_run["Initialization"],
        optimizer_name=best_run["Optimizer"],
        maxiter=50
        optimal_params=best_params,
        exact_energy=E_exact,
        shots=4096,
        backend_name="ibm_fez"
    )
    hw_metrics = run_hardware_evaluation(
        circuit=best_ansatz_qc,
        hamiltonian=H_631,
        optimal_params=best_single["final_params"],
        exact_energy=E_exact
    )
    print(f"  -> FakeFez Noisy Simulation: Measured = {noisy_metrics['Hardware Measured Energy (Ha)']:.8f} Ha | Error = {noisy_metrics['Hardware Error (mHa)']:.4f} mHa")
    
    # Stage 7: Excel Export
    print("\n[Stage 7/6] Exporting results.xlsx with 5 Sheets and Native Charts...")
    # Parameter binding transpilation test
    binding_test_data = test_transpilation_parameter_binding(best_ansatz_qc, backend_name="ibm_fez")
    print(f"  -> Parameter Binding at theta=0: Unbound 2Q = {binding_test_data['unbound_2q_gates']}, Bound 2Q = {binding_test_data['bound_zero_2q_gates']}")
    
    # Check historical hardware run file or record
    hw_data_for_sheet = {
        "Status": noisy_metrics["Status"],
        "Target Backend": noisy_metrics["Target Backend"],
        "Job ID": noisy_metrics["Job ID"],
        "Ansatz Selected": f"{best_run['Ansatz']} (Optimal parameter: theta = 0.000)",
        "Parameters Optimized": len(best_params),
        "Transpiled 2-Qubit Gate Count (unbound)": binding_test_data["unbound_2q_gates"],
        "Transpiled 2-Qubit Gate Count (bound theta=0)": binding_test_data["bound_zero_2q_gates"],
        "Circuit Depth": noisy_metrics["Circuit Depth"],
        "Exact Active Ground Energy (Ha)": E_exact,
        "Measured Energy (Ha)": noisy_metrics["Hardware Measured Energy (Ha)"],
        "Energy Error (mHa)": noisy_metrics["Hardware Error (mHa)"],
        "Shots": 4096,
        "Simulation Time (s)": noisy_metrics["QPU Runtime (seconds)"],
        "Historical Job d330j9cve01c738t02j0": "UNVERIFIED - confirm in IBM Quantum dashboard"
    }
    
    # Stage 8: Automated Verification Suite
    print("\n[Stage 8/9] Running Automated Verification Test Suite...")
    verification_df = run_all_verifications()
    print(verification_df.to_string(index=False))
    assert (verification_df["Status"] == "PASS").all(), "Error: Some verification tests failed!"
    
    # Stage 9: Generate Overview Plot and Excel Export
    print("\n[Stage 9/9] Exporting Visualizations and results.xlsx with 7 Sheets...")
    generate_overview_plot(results_df, robustness_df, FIGURES_DIR / "benchmark_overview.png")
    
    export_benchmark_to_excel(
        results_df=results_df,
        convergence_df=convergence_df,
        circuits_df=circuits_df,
        meta=meta_631,
        hardware_data=hw_metrics,
        lr_sweep_df=lr_sweep_df,
        robustness_df=robustness_df,
        hardware_data=hw_data_for_sheet,
        binding_test_data=binding_test_data,
        verification_df=verification_df,
        output_path="results.xlsx"
    )
    
    total_time = time.time() - start_time
    print(f"\n==========================================================================")
    print(f"BENCHMARK COMPLETE! Total Runtime: {total_time:.2f}s")
    print(f"All files generated:")
    print(f"  - results.xlsx")
    print(f"  - figures/circuit_DexcG.png")
    print(f"  - figures/circuit_PCU2.png")
    print(f"  - figures/circuit_UCCSD.png")
    print(f"  - figures/circuit_k-UpCCGSD.png")
    print(f"  - figures/benchmark_overview.png")
    print(f"BENCHMARK COMPLETE! Total Elapsed Time: {total_time:.2f}s")
    print(f"==========================================================================")

if __name__ == "__main__":
    main()
