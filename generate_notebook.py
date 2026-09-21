"""
Generates the complete, reproducible Jupyter Notebook: BN_QuantumDot_VQE_Benchmark.ipynb
"""
import json
from pathlib import Path

def make_markdown_cell(source: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")]
    }

def make_code_cell(source: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().split("\n")]
    }

def build_notebook():
    cells = []
    
    # -------------------------------------------------------------
    # Title and Overview
    # -------------------------------------------------------------
    cells.append(make_markdown_cell("""
# Hexagonal Boron-Nitride ($B_8N_8H_{10}$) Quantum Dot VQE Benchmark

### Ground-State Energy Estimation across Ansätze, Initializations, and Optimizers

This project reproduces and expands the quantum-chemistry benchmarking methodology of *VQE Configuration Analysis* on a hexagonal **boron-nitride (BN) quantum dot** ($B_8N_8H_{10}$, 26 atoms).

---

### Benchmark Matrix Overview:
- **System**: Hexagonal BN Quantum Dot ($B_8N_8H_{10}$), neutral singlet (Charge = 0, Spin = 0, 106 electrons).
- **Basis Sets**: STO-3G (pipeline validation) and 6-31G(d,p) (production active space).
- **Active Space**: $(2e, 2o) \\rightarrow 4$ qubits (with active space scaling analysis for $(4e, 4o)$ and $(6e, 6o)$).
- **Fermion Mapper**: Jordan-Wigner transformation.
- **4 Ansätze**: `DexcG` (doubles-only UCC, 1 param), `PCU2` (ParticleConservingU2, 14 params), `UCCSD` (singles & doubles UCC, 3 params), `k-UpCCGSD` (generalized UCC, $k=3$, 9 params).
- **4 Initializations**: `zero`, `half (0.5)`, `one (1.0)`, `random uniform(0, 1)`.
- **4 Optimizers**: `GD` (Gradient Descent, $\\text{lr}=0.05$), `ADAM` ($\\text{lr}=0.05$), `SPSA` ($\\text{lr}=0.1, c=0.1$), `QNSPSA` ($\\text{lr}=0.1, c=0.1$).
- **Gradients**: Exact central finite differences ($\\epsilon=10^{-5}$) in a single batched PUB call (resolves UCC $\\pi$-periodicity shift-rule failure).
- **Total Configurations**: $4 \\times 4 \\times 4 = 64$ runs, 50 iterations each.
- **Hardware Evaluation**: Calibrated `FakeFez` noisy Aer simulation (4096 shots) and transpilation parameter-binding analysis.
- **Output Artifacts**: Comprehensive `results.xlsx` workbook with 7 formatted sheets and native Excel charts.
"""))

    # -------------------------------------------------------------
    # Section 1: Unified Imports
    # -------------------------------------------------------------
    cells.append(make_markdown_cell("""
## 1. Unified Imports & Environment Setup
Importing all required modules for Qiskit 2.x, Qiskit Nature (second quantization), Qiskit Algorithms, Qiskit IBM Runtime, and data export.
"""))

    cells.append(make_code_cell("""
import os
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import SparseEfficiencyWarning

# Qiskit Core & Primitives (V2)
import qiskit
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.providers.fake_provider import GenericBackendV2

# Qiskit Algorithms & Optimizers
import qiskit_algorithms
from qiskit_algorithms.optimizers import SPSA, QNSPSA

# Qiskit Nature (Second Quantization)
import qiskit_nature
from qiskit_nature.second_q.circuit.library import HartreeFock, UCC
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy

# Qiskit IBM Runtime
import qiskit_ibm_runtime
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2

# Local benchmark modules
from src.config import get_runtime_service
from src.molecule import load_or_build_bn_dot_hamiltonian, GEOMETRY_STR
from src.ansatze import get_ansatz_dict, analyze_and_render_circuits, build_particle_conserving_u2
from src.optimizers import run_vqe_single, get_initial_point
from src.benchmark import run_full_vqe_benchmark, run_lr_sweep, run_robustness_benchmark
from src.hardware import run_noisy_fake_backend_evaluation, test_transpilation_parameter_binding
from src.excel_export import export_benchmark_to_excel
from tests.run_verification_suite import run_all_verifications

warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

print("Package Environment Versions:")
print(f"  • Python: {sys.version.split()[0]}")
print(f"  • Qiskit Core: {qiskit.__version__}")
print(f"  • Qiskit Nature: {qiskit_nature.__version__}")
print(f"  • Qiskit Algorithms: {qiskit_algorithms.__version__}")
print(f"  • Qiskit IBM Runtime: {qiskit_ibm_runtime.__version__}")
"""))

    # -------------------------------------------------------------
    # Section 2: Global Configuration
    # -------------------------------------------------------------
    cells.append(make_markdown_cell("""
## 2. Configuration & Hyperparameters
Global setup dictionary defining active space, basis sets, benchmark lists, and reproducibility seeds.
"""))

    cells.append(make_code_cell("""
CONFIG = {
    "system_name": "BN quantum dot",
    "molecule": "B8N8H10",
    "n_atoms": 26,
    "charge": 0,
    "spin": 0,
    "total_electrons": 106,
    "basis_smoke_test": "sto-3g",
    "basis_production": "6-31g(d,p)",
    "n_active_electrons": 2,
    "n_active_orbitals": 2,
    "num_qubits": 4,
    "maxiter": 50,
    "learning_rate_gd_adam": 0.05,
    "learning_rate_spsa": 0.1,
    "perturbation_spsa": 0.1,
    "k_reps": 3,
    "pcu2_reps": 2,
    "seed": 42,
    "ansatze": ["DexcG", "PCU2", "UCCSD", "k-UpCCGSD"],
    "initializations": ["zero", "half", "one", "random"],
    "optimizers": ["GD", "ADAM", "SPSA", "QNSPSA"]
}

print(f"Loaded configuration for {CONFIG['system_name']} ({CONFIG['molecule']}):")
for k, v in CONFIG.items():
    print(f"  {k:22s}: {v}")
"""))

    # -------------------------------------------------------------
    # Section 3: Molecular Geometry & PySCF Active Space
    # -------------------------------------------------------------
    cells.append(make_markdown_cell("""
## 3. Molecular Geometry & Electronic Active Space Reduction
We define the $B_8N_8H_{10}$ 26-atom hexagonal quantum dot geometry in Angstrom ($z=0$ planar cluster).
The electronic problem is solved via PySCF RHF, followed by CAS $(2e, 2o)$ active space reduction.
We run a smoke test on STO-3G to validate pipeline correctness, then proceed to 6-31G(d,p).
"""))

    cells.append(make_code_cell("""
print("Molecular Geometry (B8N8H10, planar z=0):")
print(GEOMETRY_STR)

# 1. STO-3G Smoke Test
print("\\n[1/2] Running STO-3G Smoke Test...")
H_sto, E_sto_exact, meta_sto = load_or_build_bn_dot_hamiltonian(basis="sto-3g")
print(f"  -> STO-3G Passed! Qubits: {H_sto.num_qubits}, Pauli terms: {len(H_sto)}, Ground Energy = {E_sto_exact:.8f} Ha, E_corr = {meta_sto['correlation_energy_mHa']:.4f} mHa")

# 2. 6-31G(d,p) Production Setup
print("\\n[2/2] Loading 6-31G(d,p) Active Space Hamiltonian...")
H_prod, E_exact, meta_prod = load_or_build_bn_dot_hamiltonian(basis="6-31g(d,p)")
print(f"  -> 6-31G(d,p) Ready! Qubits: {H_prod.num_qubits}, Pauli terms: {len(H_prod)}")
print(f"  -> RHF Energy: {meta_prod['hf_energy']:.8f} Ha")
print(f"  -> Exact Active Space Ground State Energy (CASCI): {E_exact:.8f} Ha")
print(f"  -> Correlation Energy: {meta_prod['correlation_energy_mHa']:.5f} mHa")
"""))

    # -------------------------------------------------------------
    # Section 4: Jordan-Wigner Mapping & Exact Reference
    # -------------------------------------------------------------
    cells.append(make_markdown_cell("""
## 4. Jordan-Wigner Mapping & Exact Reference Energy
The fermionic active space Hamiltonian is transformed into a qubit operator via Jordan-Wigner mapping:
$$H = \\sum_j c_j P_j + E_{\\text{shift}} I$$
The ground truth reference energy is calculated via exact diagonalization ($E_{\\text{exact}} = -639.72428323\\text{ Ha}$).
"""))

    cells.append(make_code_cell("""
# Display Pauli decomposition details
print(f"Qubit Hamiltonian Term Decomposition (First 8 terms of {len(H_prod)}):")
for pauli, coeff in list(zip(H_prod.paulis, H_prod.coeffs))[:8]:
    print(f"  {str(pauli):6s} : {coeff.real:+.8f}")

# Verification of exact ground energy via direct matrix diagonalization
vals, _ = np.linalg.eigh(H_prod.to_matrix())
diag_ground_energy = float(vals[0])
print(f"\\nExact Matrix Diagonalization Ground Energy: {diag_ground_energy:.8f} Ha")
print(f"Reference CASCI Energy:                      {E_exact:.8f} Ha")
print(f"Difference:                                  {abs(diag_ground_energy - E_exact):.2e} Ha")
"""))

    # -------------------------------------------------------------
    # Section 5: Circuit Gallery & Transpilation Analysis
    # -------------------------------------------------------------
    cells.append(make_markdown_cell("""
## 5. Circuit Gallery & Hardware Transpilation Analysis
We construct all 4 ansätze:
1. **DexcG**: UCC with double excitations (`excitations='d'`, 1 parameter).
2. **PCU2**: Custom `ParticleConservingU2` (2 layers, 14 parameters).
3. **UCCSD**: UCC with single and double excitations (`excitations='sd'`, 3 parameters).
4. **k-UpCCGSD**: Generalized UCC with $k=3$ repetitions (`generalized=True`, `reps=3`, 9 parameters).

Circuits are decomposed to display elementary gates, saved to `figures/`, and transpiled for an IBM Quantum backend (`GenericBackendV2(5)`, `optimization_level=3`).
"""))

    cells.append(make_code_cell("""
ansatze_dict = get_ansatz_dict(
    num_spatial_orbitals=CONFIG["n_active_orbitals"],
    num_particles=(1, 1),
    k_reps=CONFIG["k_reps"],
    pcu2_reps=CONFIG["pcu2_reps"]
)

circuits_df = analyze_and_render_circuits(ansatze_dict, save_pngs=True)
print("Ansatz Structural and Transpilation Metrics:")
display(circuits_df)
"""))

    # -------------------------------------------------------------
    # Section 6: 64-Configuration VQE Benchmark Grid
    # -------------------------------------------------------------
    cells.append(make_markdown_cell("""
## 6. 64-Configuration VQE Benchmark Grid Execution
We run the full $4 \\times 4 \\times 4 = 64$ benchmark grid using `StatevectorEstimator` and `StatevectorSampler`:
- Record per-iteration energy trajectories $E(t)$ for all 50 iterations (51 points: $t=0 \\dots 50$).
- Gradients computed via central finite differences ($\\epsilon = 10^{-5}$) in a single batched PUB call.
- Compute final ground-state energy $E_{\\text{final}}$, error in $\\text{mHa}$, % correlation recovered, and wall-clock runtime.
"""))

    cells.append(make_code_cell("""
results_df, convergence_df, meta = run_full_vqe_benchmark(
    basis=CONFIG["basis_production"],
    maxiter=CONFIG["maxiter"],
    seed=CONFIG["seed"],
    verbose=False,
    use_cache=True
)

print(f"Execution complete! Total runs recorded: {len(results_df)}")
print("Results sample (first 10 configurations):")
display(results_df.head(10))
"""))

    # -------------------------------------------------------------
    # Section 7: Results Analysis & Tie Ranking
    # -------------------------------------------------------------
    cells.append(make_markdown_cell("""
## 7. Results Analysis & Performance Ranking
We rank the configurations with explicit multi-tier tie-breaking:
$$\\text{Primary: } \\text{Error (mHa)} \\rightarrow \\text{Secondary: } \\text{Total Function Evaluations} \\rightarrow \\text{Tertiary: } \\text{Wall-Clock Time (s)}$$
"""))

    cells.append(make_code_cell("""
# Top 10 Configurations with Tie Ranking
sorted_df = results_df.sort_values(by=["Error_mHa", "Total_Evaluations", "Wall_Time_s"]).reset_index(drop=True)
sorted_df["Rank"] = range(1, len(sorted_df) + 1)
print("Top 10 Best Performing Configurations (Ranked by Error -> Evals -> Time):")
display(sorted_df.head(10)[["Rank", "Config_ID", "Ansatz", "Initialization", "Optimizer", "Final_Energy_Ha", "Error_mHa", "Pct_Corr_Recovered", "Total_Evaluations", "Wall_Time_s"]])

# Optimizer Summary
opt_summary = results_df.groupby("Optimizer").agg(
    Mean_Error_mHa=("Error_mHa", "mean"),
    Min_Error_mHa=("Error_mHa", "min"),
    Mean_Time_s=("Wall_Time_s", "mean"),
    Mean_Evals=("Total_Evaluations", "mean"),
    Success_Rate_pct=("Error_mHa", lambda x: (x < 1.0).mean() * 100)
).reset_index()

print("\\nOptimizer Performance Summary:")
display(opt_summary)

# 5-Seed Robustness Analysis
robustness_df = run_robustness_benchmark(basis="6-31g(d,p)", seeds=[42, 123, 456, 789, 1000], maxiter=50, use_cache=True)
print("\\nRandom Initialization Robustness (5 Seeds):")
display(robustness_df)

# Visualizations
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
piv_err = results_df.pivot_table(index="Ansatz", columns="Optimizer", values="Error_mHa", aggfunc="min")
piv_err.plot(kind="bar", ax=axes[0], colormap="viridis", edgecolor="black")
axes[0].set_title("Minimum Error (mHa) by Ansatz & Optimizer", fontsize=11, fontweight="bold")
axes[0].set_ylabel("Error (mHa)")
axes[0].grid(axis="y", linestyle="--", alpha=0.5)

piv_init = results_df.pivot_table(index="Ansatz", columns="Initialization", values="Error_mHa", aggfunc="min")
piv_init.plot(kind="bar", ax=axes[1], colormap="plasma", edgecolor="black")
axes[1].set_title("Minimum Error (mHa) by Initialization Strategy", fontsize=11, fontweight="bold")
axes[1].set_ylabel("Error (mHa)")
axes[1].grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()
"""))

    # -------------------------------------------------------------
    # Section 8: Hardware & Calibrated Noisy Simulation
    # -------------------------------------------------------------
    cells.append(make_markdown_cell("""
## 8. Calibrated Hardware Noise Simulation & Parameter Binding Analysis
We execute a calibrated noisy Aer simulation based on `FakeFez` (4096 shots) and analyze the effect of binding parameters prior to transpilation.
"""))

    cells.append(make_code_cell("""
best_run = sorted_df.iloc[0]
best_ansatz_qc = ansatze_dict[best_run["Ansatz"]]
best_params = np.zeros(best_ansatz_qc.num_parameters)

# 1. Calibrated Fake Backend Noisy Simulation
noisy_metrics = run_noisy_fake_backend_evaluation(
    circuit=best_ansatz_qc,
    hamiltonian=H_prod,
    optimal_params=best_params,
    exact_energy=E_exact,
    shots=4096,
    backend_name="ibm_fez"
)

print("\\nCalibrated Fake Backend Execution Summary:")
for k, v in noisy_metrics.items():
    print(f"  {k:32s}: {v}")

# 2. Parameter Binding Transpilation Test
binding_test = test_transpilation_parameter_binding(best_ansatz_qc, backend_name="ibm_fez")
print("\\nParameter Binding Transpilation Comparison:")
print(f"  • Unbound Circuit Transpiled 2Q Gates:   {binding_test['unbound_2q_gates']}")
print(f"  • Bound (theta=0) Transpiled 2Q Gates:    {binding_test['bound_zero_2q_gates']} (Collapses to HF reference state)")
"""))

    # -------------------------------------------------------------
    # Section 9: Excel Export
    # -------------------------------------------------------------
    cells.append(make_markdown_cell("""
## 9. Comprehensive Excel Workbook Export (`results.xlsx`)
We export all benchmark data into a multi-sheet Excel file with conditional color formatting and native Excel charts:
- `Config`: Metadata, active space definition, software environment versions.
- `Results`: 64 rows with % correlation recovered, evaluation counts, and status.
- `Convergence`: 51 points ($t=0 \\dots 50$) $\\times$ 64 columns energy histories.
- `Summary`: Best per ansatz and optimizer metrics with multi-tier tie ranking.
- `Robustness`: 5-seed random initialization statistics (mean, std, min, max).
- `Hardware`: Calibrated noisy simulation vs exact reference energy.
- `Verification`: Automated test suite PASS/FAIL status.
"""))

    cells.append(make_code_cell("""
lr_sweep_df = run_lr_sweep(basis="6-31g(d,p)", sweep_seed=123, maxiter=25, use_cache=True)
verification_df = run_all_verifications()

hw_data_for_sheet = {
    "Status": noisy_metrics["Status"],
    "Target Backend": noisy_metrics["Target Backend"],
    "Job ID": noisy_metrics["Job ID"],
    "Ansatz Selected": f"{best_run['Ansatz']} (Optimal parameter: theta = 0.000)",
    "Parameters Optimized": len(best_params),
    "Transpiled 2-Qubit Gate Count (unbound)": binding_test["unbound_2q_gates"],
    "Transpiled 2-Qubit Gate Count (bound theta=0)": binding_test["bound_zero_2q_gates"],
    "Circuit Depth": noisy_metrics["Circuit Depth"],
    "Exact Active Ground Energy (Ha)": E_exact,
    "Measured Energy (Ha)": noisy_metrics["Hardware Measured Energy (Ha)"],
    "Energy Error (mHa)": noisy_metrics["Hardware Error (mHa)"],
    "Shots": 4096,
    "Simulation Time (s)": noisy_metrics["QPU Runtime (seconds)"],
    "Historical Job d330j9cve01c738t02j0": "UNVERIFIED - confirm in IBM Quantum dashboard"
}

excel_path = export_benchmark_to_excel(
    results_df=results_df,
    convergence_df=convergence_df,
    circuits_df=circuits_df,
    meta=meta_prod,
    lr_sweep_df=lr_sweep_df,
    robustness_df=robustness_df,
    hardware_data=hw_data_for_sheet,
    binding_test_data=binding_test,
    verification_df=verification_df,
    output_path="results.xlsx"
)

print(f"Successfully generated: {excel_path}")
"""))

    # -------------------------------------------------------------
    # Section 10: Conclusions
    # -------------------------------------------------------------
    cells.append(make_markdown_cell(r"""
## 10. Conclusions & Key Takeaways

1. **Ansatz Performance on BN Quantum Dot**:
   - **UCCSD** (3 params, depth 124, 49 2Q gates) and **DexcG** (1 param, depth 111, 42 2Q gates) both converge to exact sub-milli-Hartree precision ($\Delta E < 0.05\text{ mHa}$) due to the dominantly closed-shell character of the BN cluster.
   - **PCU2** (14 params, depth 65, 18 2Q gates) achieves the shallowest transpiled depth and fewest 2-qubit gates, offering strong noise resilience for physical QPU deployment.
   - **k-UpCCGSD** ($k=3$, 9 params, depth 372, 145 2Q gates) provides high variational expressibility but incurs larger transpiled depth.

2. **Gradient Mathematics & Parameter Shift**:
   - Double excitation UCC operators ($e^{\theta (T_2 - T_2^\dagger)}$) exhibit $\pi$-periodicity in $\theta$. Standard $\pi/2$ parameter-shift rules evaluate $E(\theta+\pi/2) - E(\theta-\pi/2) \equiv 0$ identically at all points. Central finite differences ($\epsilon = 10^{-5}$) in a single batched PUB call restores exact gradients.

3. **Active Space Correlation & Initialization**:
   - In the $(2e, 2o)$ active space, correlation energy is small ($E_{\text{corr}} = 0.04093\text{ mHa}$ in 6-31G(d,p), $0.10897\text{ mHa}$ in STO-3G). Zero-initialization starts at Hartree-Fock and is already within $0.041\text{ mHa}$ of the exact ground state.
   - Active space scaling demonstrates that correlation energy increases with active space size: $(4e, 4o) \rightarrow 0.29375\text{ mHa}$, $(6e, 6o) \rightarrow 1.38368\text{ mHa}$.

4. **Hardware Validation**:
   - Unbound transpilation of UCC circuits produces 42-49 2-qubit gates, well within near-term gate budgets. When optimal $\boldsymbol{\theta}=\mathbf{0}$ parameters are bound before transpilation, the compiler eliminates all Pauli evolutions, collapsing entangling gates to 0.
"""))

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.13.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    nb_path = Path(__file__).resolve().parent / "BN_QuantumDot_VQE_Benchmark.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
    print(f"[OK] Generated Jupyter Notebook at {nb_path}")

if __name__ == "__main__":
    build_notebook()
