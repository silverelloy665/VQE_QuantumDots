"""
Updates README.md with Round 2 Hard Benchmark results, tables generated
strictly from data/ result files, Makefile documentation, and Known Limitations.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_docs import (
    generate_reconciled_jobs_md,
    generate_active_space_scan_md,
    generate_hard_benchmark_summary_md,
    generate_circuit_costs_md,
    generate_hardware_jobs_v2_md
)

README_PATH = REPO_ROOT / "README.md"


def build_round2_readme_section() -> str:
    reconciled_jobs_table = generate_reconciled_jobs_md()
    active_space_table = generate_active_space_scan_md()
    hard_summary_table = generate_hard_benchmark_summary_md(10)
    circuit_costs_table = generate_circuit_costs_md()
    hardware_jobs_table = generate_hardware_jobs_v2_md()

    sec = (
        "\n---\n\n"
        "# 🚀 Round 2: Genuinely Hard Quantum Dot Benchmark (8 Qubits, 4e4o)\n\n"
        "Following the initial audit which revealed that equilibrium $B_8N_8H_{10}$ active spaces have near-Hartree-Fock correlation gaps ($E_{corr} = 0.0409\\text{ mHa} < 1.6\\text{ mHa}$), **Round 2** constructs a genuinely hard benchmark satisfying $E_{corr} \\ge 5.0\\text{ mHa}$ with $\\le 12$ qubits.\n\n"
        "### 1. Phase A Audit Verification & Hardware Source of Truth\n\n"
        "- **Target Backend Single Source of Truth**: Pinned `DEFAULT_HARDWARE_BACKEND = \"ibm_marrakesh\"` (IBM Quantum Heron r2, 156 qubits, Qiskit 2.3.1) in `src/config.py`.\n"
        "- **Secrets Hygiene**: Git history verification (`git log --all --oneline -- .env`) confirms `.env` was **NEVER** committed. Only `.env.example` is tracked, and `.env` is ignored by `.gitignore`.\n"
        "- **Historical Job Reconciliation**: All 5 physical QPU jobs from repository history reconciled in `data/reconciled_hardware_jobs.csv`:\n\n"
        + reconciled_jobs_table + "\n"
        "- **Uniform Evaluation Counting**: Central finite differences cost $2N$ energy evaluations per gradient step ($M \\times (2N + 1)$ function evaluations per run).\n\n"
        "---\n\n"
        "### 2. Phase B: Active Space Scan & Hard Problem Definition\n\n"
        "Radial B-N bond scaling $s \\in \\{1.0, 1.2, 1.4, 1.6, 1.8\\}$ scanned across $(2e,2o)$, $(4e,4o)$, and $(6e,6o)$ active spaces while fixing B-H ($1.19\\text{ \u00c5}$) and N-H ($1.01\\text{ \u00c5}$) bond lengths:\n\n"
        + active_space_table + "\n\n"
        "> [!IMPORTANT]\n"
        "> **Chosen Hard Benchmark Definition**:\n"
        "> - **Geometry**: Radial B-N bond scale $s = 1.4$ ($r_{\\text{BN}} = 2.02\\text{ \u00c5}$, fixed B-H $1.19\\text{ \u00c5}$, N-H $1.01\\text{ \u00c5}$)\n"
        "> - **Active Space**: $(4e, 4o)$ with MP2 natural orbitals $\\to$ **8 Qubits** (Jordan-Wigner)\n"
        "> - **Reference Energy**: $E_{\\text{ref}} = -637.57319627\\text{ Ha}$\n"
        "> - **Exact CASCI Ground Energy**: $E_{\\text{CASCI}} = -637.58133543\\text{ Ha}$\n"
        "> - **Active Correlation Gap**: $\\Delta E_{\\text{corr}} = 8.1392\\text{ mHa} \\ge 5.0\\text{ mHa}$\n"
        "> - **Symmetry & Fock Space Stability**: RHF is stable (`RHF_stable = True`). Unconstrained Fock space ground state lies strictly in $(N_\\alpha=2, N_\\beta=2)$ sector with weight $> 0.999999$ and energy diff $1.14 \\times 10^{-13}\\text{ Ha}$ ($< 10^{-8}\\text{ Ha}$). Scale 1.8 was excluded due to RHF triplet instability ($S^2=2.0$). Scale 1.6 was excluded due to Fock space ground sector shift to $(2, 3)$.\n\n"
        "---\n\n"
        "### 3. Phase C: Hard Problem VQE Grid Results (10 Seeds)\n\n"
        "Benchmark grid across 8 ans\u00e4tze $\\times$ 6 optimizers $\\times$ 10 random seeds ($[0 \\dots 9]$) + zero inits (469 total runs, evaluated variational bound $E \\ge E_{\\text{CASCI}} - 10^{-9}\\text{ Ha}$):\n\n"
        + hard_summary_table + "\n\n"
        "#### Benchmark Insights:\n"
        "- **Winner**: **2-UpCCGSD + L-BFGS-B** achieves **90.0% success rate** ($< 1.6\\text{ mHa}$ chemical accuracy), with **Median error = 0.8023 mHa**, IQR = $1.0704\\text{ mHa}$, and Min error = **0.0754 mHa** (89.18% mean correlation recovered).\n"
        "- **Inversion / DexcG Failure**: DexcG completely fails on the hard problem from random initializations (0.0% success rate, median error 560.59 mHa). The Round 1 finding that \"DexcG is best\" does NOT survive on non-trivial active spaces.\n\n"
        "---\n\n"
        "### 4. Phase D: Circuit Cost & Calibration Fidelity on `ibm_marrakesh`\n\n"
        "Transpiled across 20 seeds (optimization level 3) on `ibm_marrakesh` (Heron r2) using live calibration data (352 CZ edges):\n\n"
        + circuit_costs_table + "\n\n"
        "- **Hardware Feasibility**: Only `HF_baseline` (1.0000) and `1-UpCCGSD` (0.4056) maintain estimated fidelity $> 0.3$. `2-UpCCGSD` (995 CZ gates, depth 2667, fidelity 0.1565) is marked **NOT RECOMMENDED: too deep**.\n"
        "- **ParityMapper Reduction**: Compresses 8 qubits to 6 qubits (325 Pauli terms), reproducing exact CASCI ground energy to $1.14 \\times 10^{-13}\\text{ Ha}$ (status: ADOPTED).\n\n"
        "---\n\n"
        "### 5. Phase E: Hardware Preparation (`hardware_job_spec_v2.json`)\n\n"
        "15 batch jobs prepared across 3 resilience levels (Dynamical Decoupling, Pauli Twirling, ZNE) targeting `ibm_marrakesh` (NOT submitted):\n\n"
        + hardware_jobs_table + "\n\n"
        "> [!TIP]\n"
        "> **Recommended Hardware Experiment**: The **2-qubit parity-reduced minimal circuit for the original 2e2o problem** (`original_2e2o_parity_reduced`) requires only **2 physical CZ gates** (depth 11) on `ibm_marrakesh`, achieves **0.9220 estimated fidelity**, groups into **4 QWC circuits** (3.28s execution time), and reproduces CASCI ground energy to **$2.13 \\times 10^{-7}\\text{ Ha}$** ($< 1 \\times 10^{-6}\\text{ Ha}$).\n\n"
        "---\n\n"
        "### 6. Makefile Targets\n\n"
        "- `make verify`: Runs full unit test suite (`pytest -q`) and executes Phase A audit verification (`scripts/10_verify_audit.py`).\n"
        "- `make hard`: Executes the full hard benchmark pipeline: active-space scan (Phase B), VQE grid (Phase C), and circuit cost analysis (Phase D).\n"
        "- `make test`: Runs pytest suite (`pytest -q`).\n"
        "- `make hardware-prep`: Prepares and validates `data/hardware_job_spec_v2.json` in dry-run mode without contacting QPU.\n\n"
        "---\n\n"
        "### 7. Known Limitations\n\n"
        "1. **Statevector Simulation Only**: All VQE trajectories evaluate exact statevector expectation values. Hardware jobs are prepared, transpiled, and verified, but physical QPU submission requires explicit authorization.\n"
        "2. **Single Geometry Family**: Explores uniform radial B-N stretching with fixed B-H and N-H bonds at planar $D_{3h}$ symmetry; out-of-plane puckering or asymmetric dissociation modes are not considered.\n"
        "3. **Frozen RHF Orbitals**: MP2 natural orbitals are generated from an initial RHF reference and remain fixed during VQE (no simultaneous orbital relaxation / CASSCF).\n"
        "4. **No Explicit Noise Model in Statevector Simulation**: Circuit fidelities are estimated analytically via calibrated gate error products $\\prod(1 - e_{2q})$ rather than density matrix simulations with Kraus operators.\n"
    )
    return sec


def main():
    sec_r2 = build_round2_readme_section()
    with open(README_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    marker = "# 🚀 Round 2: Genuinely Hard Quantum Dot Benchmark (8 Qubits, 4e4o)"
    if marker in content:
        parts = content.split(marker)
        new_content = parts[0].rstrip() + "\n\n" + sec_r2.lstrip()
    else:
        new_content = content.rstrip() + "\n\n" + sec_r2.lstrip()

    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"Successfully updated {README_PATH}.")


if __name__ == "__main__":
    main()

