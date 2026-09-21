# Hexagonal Boron-Nitride ($B_8N_8H_{10}$) Quantum Dot VQE Benchmark

A mathematically rigorous and reproducible quantum chemistry benchmarking project implementing the Variational Quantum Eigensolver (VQE) to compute the ground-state electronic energy of a **hexagonal boron-nitride ($B_8N_8H_{10}$, 26 atoms) quantum dot** ("BN quantum dot").

This repository benchmarks across a full factorial grid:
**4 Ansätze $\times$ 4 Initializations $\times$ 4 Optimizers = 64 VQE Configurations (50 iterations each)**, with learning rate sweeps, 5-seed random initialization robustness testing, active-space scaling analysis, and calibrated noisy quantum hardware simulation.

---

## 🔬 System Overview: Hexagonal BN Quantum Dot

- **Stoichiometry**: $B_8 N_8 H_{10}$ (26 atoms: 8 Boron, 8 Nitrogen, 10 Hydrogen)
- **Geometry**: Planar hexagonal lattice ($z = 0$, $D_{3h}$ core symmetry, hydrogen-passivated boundary)
- **Electronic Structure**: Neutral singlet ($Q = 0, S = 0$, 106 total electrons)
- **Active Space**: $(2e, 2o)$ active space around the HOMO/LUMO frontier
- **Qubit Encoding**: 4 qubits via Jordan-Wigner transformation
- **Hamiltonian**: 27 Pauli operator terms ($I$, $Z_i$, $Z_i Z_j$, $X_i X_j Y_k Y_l$, etc.)

### Reference Electronic Energies (PySCF 2.14.0)
| Basis Set / Active Space | RHF Energy (Hartree) | Exact Active CASCI (Hartree) | Active Correlation Energy $E_{\text{corr}}$ |
| :--- | :--- | :--- | :--- |
| **STO-3G $(2e, 2o)$** (Smoke Test) | $-631.74167448\text{ Ha}$ | **$-631.74178346\text{ Ha}$** | $0.10897\text{ mHa}$ |
| **6-31G(d,p) $(2e, 2o)$** (Production) | $-639.72424230\text{ Ha}$ | **$-639.72428323\text{ Ha}$** | $0.04093\text{ mHa}$ |
| **6-31G(d,p) $(4e, 4o)$** (Scaled) | $-639.72424230\text{ Ha}$ | **$-639.72453605\text{ Ha}$** | $0.29375\text{ mHa}$ |
| **6-31G(d,p) $(6e, 6o)$** (Scaled) | $-639.72424230\text{ Ha}$ | **$-639.72562598\text{ Ha}$** | $1.38368\text{ mHa}$ |
| **6-31G(d,p) B3LYP $(2e, 2o)$** | $-639.72424230\text{ Ha}$ | **$-639.72428323\text{ Ha}$** | $0.04093\text{ mHa}$ |

> **Note on Active Space Correlation**: In the $(2e, 2o)$ frontier active space, the Hartree-Fock state $|0101\rangle$ is already within $0.041\text{ mHa}$ of the exact ground state due to the strong closed-shell ionic bonding of BN quantum dots. Multi-orbital active space scaling shows that $E_{\text{corr}}$ grows systematically from $0.041\text{ mHa} \to 0.294\text{ mHa} \to 1.384\text{ mHa}$ as $(4e, 4o)$ and $(6e, 6o)$ active spaces incorporate dynamic correlation.

---

## 📐 Benchmark Factorial Grid ($4 \times 4 \times 4 = 64$ Configurations)

### 1. Ansätze (4)
1. **DexcG**: UCC with double excitation operators (`excitations='d'`, 1 variational parameter).
2. **PCU2**: Custom `ParticleConservingU2` with 2 layers preserving particle number $\eta=2$ (14 variational parameters).
3. **UCCSD**: Unitary Coupled Cluster with Singles and Doubles (`excitations='sd'`, 3 variational parameters).
4. **k-UpCCGSD**: Generalized UCC with $k=3$ repetitions (`generalized=True`, `reps=3`, 9 variational parameters).

### 2. Parameter Initializations (4)
1. **Zero**: $\boldsymbol{\theta}_0 = \mathbf{0}$ (corresponds exactly to the Hartree-Fock state for UCC ansätze).
2. **Half**: $\boldsymbol{\theta}_0 = 0.5 \cdot \mathbf{1}$.
3. **One**: $\boldsymbol{\theta}_0 = 1.0 \cdot \mathbf{1}$.
4. **Random**: $\boldsymbol{\theta}_0 \sim \mathcal{U}(0, 1)$ with fixed seed $42$.

### 3. Classical Optimizers (4)
1. **GD**: Gradient Descent ($\text{lr}=0.05$).
2. **ADAM**: Adaptive Moment Estimation ($\text{lr}=0.05, \beta_1=0.9, \beta_2=0.999$).
3. **SPSA**: Simultaneous Perturbation Stochastic Approximation ($\text{lr}=0.1, c=0.1$).
4. **QNSPSA**: Quantum Natural SPSA with state fidelity quantum metric tensor ($\text{lr}=0.1, c=0.1$).

---

## 🧮 Gradient Computation & Parameter-Shift Failure Correction

UCC excitation operators under Jordan-Wigner transformation contain Pauli string generators $G_k$ (e.g. $X_0 X_1 Y_2 X_3$). The energy response $E(\theta) = \langle \text{HF} | e^{-\theta G} H e^{\theta G} | \text{HF} \rangle$ is strictly $\pi$-periodic in $\theta$, meaning $E(\theta + \pi/2) - E(\theta - \pi/2) \equiv 0$ identically everywhere.

Consequently, standard single-generator $\pi/2$ parameter-shift rules yield a gradient of **0.000000** at all test points. 

**Correction Implemented**: Vectorized **central finite differences** ($\epsilon = 10^{-5}$) evaluating all $2N$ parameter points simultaneously in a **single batched PUB call**:
$$\frac{\partial E(\boldsymbol{\theta})}{\partial \theta_i} \approx \frac{E(\boldsymbol{\theta} + \epsilon \mathbf{e}_i) - E(\boldsymbol{\theta} - \epsilon \mathbf{e}_i)}{2\epsilon}$$
This provides numerically exact, monotonically descending gradients for GD and ADAM across all ansätze.

---

## 📊 Circuit Structural & Hardware Transpilation Metrics

Circuits transpiled against IBM Quantum Heron / Eagle architecture (`GenericBackendV2(5)`, `optimization_level=3`):

| Ansatz | Variational Parameters | Native 2Q Gates | Raw Circuit Depth | Transpiled Circuit Depth | Transpiled 2Q Gates | Transpiled 1Q Gates |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DexcG** | 1 | 8 | 28 | 111 | 42 | 69 |
| **PCU2** | 14 | 6 | 12 | 65 | 18 | 47 |
| **UCCSD** | 3 | 10 | 32 | 124 | 49 | 75 |
| **k-UpCCGSD** ($k=3$) | 9 | 30 | 96 | 372 | 145 | 227 |

> **Hardware Feasibility**: All transpiled circuits require $\le 145$ two-qubit gates, well within physical QPU coherence limits ($< 300$ 2Q gates). `PCU2` achieves the shallowest transpiled depth (65) and fewest 2Q gates (18).

---

## 🏆 Top Configurations & Multi-Tier Tie Ranking

Configurations ranked with explicit multi-tier tie breaking:
$$\text{Primary: } \text{Error (mHa)} \longrightarrow \text{Secondary: } \text{Total Function Evaluations} \longrightarrow \text{Tertiary: } \text{Wall-Clock Time (s)}$$

| Rank | Ansatz | Initialization | Optimizer | Final Energy (Ha) | Error (mHa) | % Corr Recovered | Total Evals | Wall Time (s) |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **UCCSD** | **random** | **GD** | **$-639.72428323$** | **$1.14 \times 10^{-10}$** | $100.00\%$ | 351 | $6.10\text{ s}$ |
| **2** | **UCCSD** | **zero** | **GD** | **$-639.72428323$** | **$1.14 \times 10^{-10}$** | $100.00\%$ | 351 | $6.65\text{ s}$ |
| **3** | **UCCSD** | **one** | **GD** | **$-639.72428323$** | **$2.27 \times 10^{-10}$** | $100.00\%$ | 351 | $6.96\text{ s}$ |
| **4** | **UCCSD** | **half** | **GD** | **$-639.72428323$** | **$3.41 \times 10^{-10}$** | $100.00\%$ | 351 | $6.33\text{ s}$ |
| **5** | **k-UpCCGSD** | **zero** | **GD** | **$-639.72428321$** | **$2.04 \times 10^{-05}$** | $99.95\%$ | 951 | $38.08\text{ s}$ |
| **6** | **DexcG** | **zero** | **QNSPSA** | **$-639.72428301$** | **$2.13 \times 10^{-04}$** | $99.48\%$ | 179 | $4.97\text{ s}$ |
| **7** | **DexcG** | **half** | **GD** | **$-639.72428301$** | **$2.13 \times 10^{-04}$** | $99.48\%$ | 151 | $1.34\text{ s}$ |
| **8** | **DexcG** | **zero** | **GD** | **$-639.72428301$** | **$2.13 \times 10^{-04}$** | $99.48\%$ | 151 | $1.27\text{ s}$ |
| **9** | **DexcG** | **zero** | **SPSA** | **$-639.72428301$** | **$2.13 \times 10^{-04}$** | $99.48\%$ | 153 | $1.26\text{ s}$ |
| **10** | **DexcG** | **random** | **GD** | **$-639.72428301$** | **$2.13 \times 10^{-04}$** | $99.48\%$ | 151 | $1.32\text{ s}$ |

### Optimizer Performance Summary
| Optimizer | Mean Error (mHa) | Min Error (mHa) | Mean Evals | Mean Time (s) | Success Rate (< 1 mHa) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ADAM** | $0.898\text{ mHa}$ | $8.30 \times 10^{-3}\text{ mHa}$ | 726.0 | $12.38\text{ s}$ | **$81.25\%$** |
| **GD** | $3.009\text{ mHa}$ | $1.14 \times 10^{-10}\text{ mHa}$ | 726.0 | $12.62\text{ s}$ | **$75.00\%$** |
| **QNSPSA** | $1.004\text{ mHa}$ | $2.13 \times 10^{-4}\text{ mHa}$ | 179.0 | $11.59\text{ s}$ | **$75.00\%$** |
| **SPSA** | $9.145\text{ mHa}$ | $2.13 \times 10^{-4}\text{ mHa}$ | **153.0** | **$2.96\text{ s}$** | $68.75\%$ |

---

## 🎲 Random Initialization Robustness (5 Fixed Seeds)

Evaluated across seeds $[42, 123, 456, 789, 1000]$:

| Ansatz | Optimizer | Mean Error (mHa) | Std Error (mHa) | Min Error (mHa) | Max Error (mHa) | Mean % Corr |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **DexcG** | **GD** | $0.000213$ | $0.000000$ | $0.000213$ | $0.000213$ | $99.48\%$ |
| **DexcG** | **ADAM** | $0.008304$ | $0.000000$ | $0.008304$ | $0.008304$ | $79.71\%$ |
| **DexcG** | **SPSA** | $0.000499$ | $0.000244$ | $0.000213$ | $0.000880$ | $98.78\%$ |
| **DexcG** | **QNSPSA** | $0.000346$ | $0.000109$ | $0.000213$ | $0.000480$ | $99.15\%$ |
| **PCU2** | **GD** | $1.527339$ | $2.842777$ | $0.000213$ | $7.135843$ | $-3632.17\%$ |
| **PCU2** | **ADAM** | $0.155459$ | $0.231267$ | $0.008304$ | $0.618080$ | $-279.79\%$ |
| **PCU2** | **SPSA** | $0.003975$ | $0.003290$ | $0.000806$ | $0.009493$ | $90.29\%$ |
| **PCU2** | **QNSPSA** | $0.001633$ | $0.001150$ | $0.000343$ | $0.003328$ | $96.01\%$ |
| **UCCSD** | **GD** | $0.000000$ | $0.000000$ | $0.000000$ | $0.000000$ | $100.00\%$ |
| **UCCSD** | **ADAM** | $0.008304$ | $0.000000$ | $0.008304$ | $0.008304$ | $79.71\%$ |
| **UCCSD** | **SPSA** | $0.000624$ | $0.000305$ | $0.000213$ | $0.000966$ | $98.48\%$ |
| **UCCSD** | **QNSPSA** | $0.000349$ | $0.000164$ | $0.000213$ | $0.000641$ | $99.15\%$ |
| **k-UpCCGSD** | **GD** | $10.508688$ | $15.526233$ | $0.000000$ | $39.467431$ | $-25575.52\%$ |
| **k-UpCCGSD** | **ADAM** | $3.418047$ | $2.709325$ | $0.008304$ | $6.974492$ | $-8251.34\%$ |
| **k-UpCCGSD** | **SPSA** | $36.574488$ | $46.852924$ | $0.000632$ | $120.218556$ | $-89255.43\%$ |
| **k-UpCCGSD** | **QNSPSA** | $3.998492$ | $2.842792$ | $0.000788$ | $7.674996$ | $-9669.75\%$ |

---

## ⚡ Calibrated Quantum Device Simulation & Parameter Binding Analysis

Hardware evaluation is conducted with a genuine calibrated noisy Aer simulation based on `FakeFez` (156-qubit Heron architecture) and parameter-binding transpilation analysis:

| Parameter / Metric | Value |
| :--- | :--- |
| **Noise Model** | `FakeFez` calibrated noise model |
| **Shots** | 4,096 |
| **Ansatz Evaluated** | `UCCSD` ($\boldsymbol{\theta}^* = \mathbf{0}$) |
| **Exact CASCI Ground Energy** | **$-639.72428323\text{ Ha}$** |
| **Noisy Aer Measured Energy** | **$-639.72838844\text{ Ha}$** |
| **Hardware Noise Error** | **$4.1052\text{ mHa}$** |
| **Historical Job `d330j9cve01c738t02j0`** | `UNVERIFIED - confirm in IBM Quantum dashboard` |
| **Unbound Transpiled 2Q Gates** | 49 |
| **Bound ($\boldsymbol{\theta}=\mathbf{0}$) Transpiled 2Q Gates** | **0** (collapses to reference Hartree-Fock state) |

---

## 📁 Repository Structure & Deliverables

```
VQE_QuantumDots/
├── BN_QuantumDot_VQE_Benchmark.ipynb # Executed interactive Jupyter Notebook
├── results.xlsx                      # Comprehensive 7-sheet workbook with native charts
├── README.md                         # Benchmark report and findings
├── CHANGELOG.md                      # Audit of fixes and improvements
├── generate_notebook.py              # Notebook builder script
├── run_benchmark.py                  # Headless benchmark runner
├── run_live_hardware_job.py          # Real IBM Quantum hardware job launcher
├── .env.example                      # IBM Quantum token template
├── data/
│   ├── bn_dot_631gdp.json            # 6-31G(d,p) (2e,2o) integrals & CASCI reference
│   ├── bn_dot_631gdp_4e4o.json       # 6-31G(d,p) (4e,4o) integrals & CASCI reference
│   ├── bn_dot_631gdp_6e6o.json       # 6-31G(d,p) (6e,6o) integrals & CASCI reference
│   ├── bn_dot_631gdp_b3lyp.json      # 6-31G(d,p) B3LYP integrals & CASCI reference
│   ├── bn_dot_sto3g.json             # STO-3G (2e,2o) integrals & CASCI reference
│   ├── benchmark_results_6-31gd_p.json # Full 64-run benchmark metrics & trajectories
│   ├── lr_sweep_results.json         # 24-point learning rate tuning sweep data
│   └── robustness_results.json       # 80-run 5-seed robustness benchmark data
├── figures/
│   ├── circuit_DexcG.png             # Decomposed circuit diagram (DexcG)
│   ├── circuit_PCU2.png              # Decomposed circuit diagram (PCU2)
│   ├── circuit_UCCSD.png             # Decomposed circuit diagram (UCCSD)
│   ├── circuit_k-UpCCGSD.png         # Decomposed circuit diagram (k-UpCCGSD)
│   └── benchmark_overview.png        # 4-panel overview comparison chart
├── src/
│   ├── __init__.py
│   ├── config.py                     # IBM Quantum credentials & service loader
│   ├── molecule.py                   # Geometry, active spaces, and exact solver
│   ├── compute_pyscf.py              # PySCF electron integral driver
│   ├── ansatze.py                    # 4 Ansatz generators and circuit transpiler
│   ├── optimizers.py                 # Vectorized central finite-difference optimizers
│   ├── benchmark.py                  # Full 64-run grid executor & sweeps
│   ├── hardware.py                   # Calibrated FakeFez noisy Aer runner & binding test
│   └── excel_export.py               # openpyxl 7-sheet workbook & chart exporter
└── tests/
    ├── conftest.py
    ├── run_verification_suite.py     # Verification runner for Excel export
    ├── test_ansatze.py               # Parameter count and HF reproduction tests
    ├── test_b3lyp_active_space.py    # DFT orbital active space tests
    ├── test_conservation.py          # Particle number & Sz conservation tests
    ├── test_geometry.py              # Molecular geometry and bond length tests
    ├── test_gradients.py             # Vectorized finite-difference gradient tests
    ├── test_hamiltonian.py           # Hamiltonian diagonalization tests
    ├── test_pyscf_reference.py       # PySCF RHF & CASCI reference tests
    └── test_reproducibility.py       # Deterministic seed reproducibility tests
```

---

## 📗 Excel Workbook Specification (`results.xlsx`)

The generated `results.xlsx` workbook contains **7 formatted sheets** with zero fabricated numbers:
1. **`Config`**: Metadata, active space definition, software environment versions, and LR sweep results.
2. **`Results`**: Full 64-configuration table with % correlation recovered, evaluation counts, and 3-color conditional formatting.
3. **`Convergence`**: Complete 51-point ($t=0 \dots 50$) energy trajectories for all 64 configurations.
4. **`Summary`**: Best per ansatz, top configurations with multi-tier tie ranking, optimizer overview, circuit metrics, and 4 native Excel charts.
5. **`Robustness`**: 5-seed random initialization statistics (mean, std, min, max, % correlation).
6. **`Hardware`**: Calibrated FakeFez noisy Aer simulation (4096 shots) and parameter binding analysis.
7. **`Verification`**: Automated verification test suite results with PASS/FAIL status.

---

## 🧪 Verification & Testing

Run the automated test suite across all 8 modules (15 unit tests):
```bash
pytest -v tests/
```

All 15 tests pass deterministically.

---

## 🚀 Execution Instructions

### 1. Execute Jupyter Notebook
```bash
python generate_notebook.py
python -m jupyter nbconvert --to notebook --execute --inplace BN_QuantumDot_VQE_Benchmark.ipynb
```

### 2. Run Headless Benchmark
```bash
python run_benchmark.py
```
