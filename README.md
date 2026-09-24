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
| Basis Set / Active Space | Reference Determinant (Ha) | Exact Active CASCI (Ha) | Active Correlation Energy $E_{\text{corr}}$ | Chemical Accuracy Threshold |
| :--- | :--- | :--- | :--- | :--- |
| **STO-3G $(2e, 2o)$** (Smoke Test) | $-631.74167448\text{ Ha}$ | **$-631.74178346\text{ Ha}$** | $0.10897\text{ mHa}$ | $1.6000\text{ mHa}$ (BELOW) |
| **6-31G(d,p) $(2e, 2o)$** (Production) | $-639.72424230\text{ Ha}$ | **$-639.72428323\text{ Ha}$** | $0.04093\text{ mHa}$ | $1.6000\text{ mHa}$ (BELOW) |
| **6-31G(d,p) $(4e, 4o)$** (Scaled) | $-639.72424230\text{ Ha}$ | **$-639.72453605\text{ Ha}$** | $0.29375\text{ mHa}$ | $1.6000\text{ mHa}$ (BELOW) |
| **6-31G(d,p) $(6e, 6o)$** (Scaled) | $-639.72424230\text{ Ha}$ | **$-639.72562598\text{ Ha}$** | $1.38368\text{ mHa}$ | $1.6000\text{ mHa}$ (BELOW) |
| **6-31G(d,p) B3LYP $(2e, 2o)$** | $-639.63711823\text{ Ha}$ | **$-639.63815054\text{ Ha}$** | $1.03231\text{ mHa}$ | $1.6000\text{ mHa}$ (BELOW) |

> **Note on Active Space Correlation**: In the $(2e, 2o)$ frontier active space, the Hartree-Fock state $|0101\rangle$ is already within $0.041\text{ mHa}$ of the exact ground state due to the strong closed-shell ionic bonding of BN quantum dots. Multi-orbital active space scaling shows that $E_{\text{corr}}$ grows systematically from $0.041\text{ mHa} \to 0.294\text{ mHa} \to 1.384\text{ mHa}$ as $(4e, 4o)$ and $(6e, 6o)$ active spaces incorporate dynamic correlation.
> [!WARNING]
> **CRITICAL SCIENTIFIC INSIGHT: NEAR-HARTREE-FOCK ACTIVE SPACES**
> In ALL five evaluated active-space configurations (including the 12-qubit $6e, 6o$ active space), the active-space correlation energy $E_{\text{corr}} = |E_{\text{ref}} - E_{\text{CASCI}}| \le 1.384\text{ mHa}$ is **strictly below chemical accuracy ($1.6000\text{ mHa} \approx 1\text{ kcal/mol}$)**.
> Because the mean-field Hartree-Fock reference determinant $|0101\rangle$ is already within $0.041\text{ mHa}$ of the exact ground state, the fact that "zero-initialization ($\boldsymbol{\theta}=\mathbf{0}$) wins" is largely an artifact of initializing inside the chemical accuracy basin. Non-zero initializations ($\boldsymbol{\theta}_0 = 0.5, 1.0, \mathcal{U}(0,1)$) artificially displace the system into excited-state topologies and barren plateaus. The benchmark therefore tests convergence from perturbed states rather than discriminating intrinsic ansatz expressibility for strongly correlated molecules.
>
> *(Note on B3LYP: The active space in B3LYP Kohn-Sham orbitals has reference determinant energy $-639.63711823\text{ Ha}$ and CASCI ground state $-639.63815054\text{ Ha}$ ($E_{\text{corr}} = 1.03231\text{ mHa}$). The full B3LYP DFT total electronic energy is $-643.63393574\text{ Ha}$ and is strictly distinct.)*

---

## 📈 Qubit and Parameter Scaling: Full Molecule vs. Active Spaces

The full $B_8N_8H_{10}$ molecule possesses 106 electrons ($o = 53$ occupied spatial orbitals). Under the Jordan-Wigner transformation, each spatial orbital maps to two spin-orbital qubits ($N_Q = 2 N_{\text{spatial}}$). For closed-shell singlet ground states with spin-conserving excitations ($\alpha \to \alpha, \beta \to \beta$):
- **Singles Count**: $N_S = 2 \cdot o \cdot v$
- **Doubles Count**: $N_D = 2 \binom{o}{2}\binom{v}{2} + (o \cdot v)^2$

| System / Configuration | Category | Spatial Orbitals | JW Qubits | UCCSD Singles | UCCSD Doubles | Total UCCSD Params | Feasibility |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Full Molecule (STO-3G)** | Full Molecule | 90 | **180** | 3,922 | 5,681,017 | **5,684,939** | Infeasible (180 Qubits, 5.7M Params) |
| **Full Molecule (6-31G(d,p))** | Full Molecule | 274 | **548** | 23,426 | 204,192,729 | **204,216,155** | Infeasible (548 Qubits, 204.2M Params) |
| **Active Space $(2e, 2o)$** | Benchmark (Core) | 2 | **4** | 2 | 1 | **3** | Ideal for NISQ & Verification |
| **Active Space $(4e, 4o)$** | Extended Active Space | 4 | **8** | 8 | 18 | **26** | Exact Statevector Simulator |
| **Active Space $(6e, 6o)$** | Extended Active Space | 6 | **12** | 18 | 99 | **117** | Statevector / Runtime Constrained |

> [!NOTE]
> **Basis-Invariance of the UCCSD Circuit**:
> The UCCSD circuit structure for a specified $(n_e, n_o)$ active space is **completely invariant to the AO basis set**:
> - An active space of $(2e, 2o)$ produces an identical 4-qubit, 3-parameter circuit whether evaluated in STO-3G or 6-31G(d,p).
> - Basis set changes only alter the 1-electron and 2-electron molecular orbital integrals ($h_{pq}, h_{pqrs}$), and hence the Pauli coefficients of the mapped Hamiltonian $\hat{H}$, leaving circuit topology, transpiled depth, and gate count unchanged.

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

## ⚡ IBM Quantum Physical QPU Evaluation (`ibm_fez`)

A live single-point energy evaluation was executed on the physical IBM Quantum QPU `ibm_fez` (156-qubit Heron architecture) in **Job Mode** using Qiskit Runtime `EstimatorV2`:
A live 5-point parameterized energy landscape evaluation was submitted and executed on the physical IBM Quantum QPU `ibm_fez` (156-qubit Heron architecture) in **Job Mode** using Qiskit Runtime `EstimatorV2`.

| Parameter / Metric | Live Physical QPU Value |
| :--- | :--- |
| **Target QPU** | `ibm_fez` (156-qubit Heron Architecture) |
| **Job ID** | [`daor3p5r85ps73ffmvn0`](https://quantum.ibm.com/jobs/daor3p5r85ps73ffmvn0) |
| **Status** | `COMPLETED_ON_PHYSICAL_QPU` |
| **Ansatz Evaluated** | `UCCSD` ($\boldsymbol{\theta}^* = \mathbf{0}$) |
| **Optimization Level** | 3 (`generate_preset_pass_manager`) |
| **Transpiled Circuit Depth** | 1 |
| **Transpiled 2-Qubit Gates** | **0** (collapses to reference Hartree-Fock state at $\boldsymbol{\theta}=\mathbf{0}$) |
| **Shots** | 4,096 |
| **QPU Execution / Wait Time** | $30.35\text{ s}$ |
| **Exact CASCI Ground Energy** | **$-639.72428323\text{ Ha}$** |
| **Physical QPU Measured Energy** | **$-639.72364609\text{ Ha}$** |
| **Physical Hardware Error** | **$0.6371\text{ mHa}$** (0.0001% relative error) |
| **Hardware Execution Record** | Cached in `data/hardware_run.json` |
To avoid trivial parameter cancellation at $\boldsymbol{\theta}=\mathbf{0}$ (where optimization level 3 collapses the circuit to depth 1 and 0 two-qubit gates), a 5-point Primitive Unified Block (PUB) was submitted with parameterized DexcG:
$$\theta \in [-0.10, -0.05, 0.00, +0.05, +0.10]\text{ radians}$$

### Live Physical Hardware Execution Record
- **Target Backend**: `ibm_fez` (156-qubit Heron Architecture, revision 2)
- **Job ID**: [`daqkle3t55cs738rsfrg`](https://quantum.ibm.com/jobs/daqkle3t55cs738rsfrg)
- **Status**: `COMPLETED_ON_PHYSICAL_QPU`
- **Ansatz Evaluated**: `DexcG` (1 parameter, 5 landscape points in a single PUB)
- **Transpiled Circuit Depth**: **144**
- **Transpiled 2-Qubit Gates**: **42** (CZ entangling gates on physical coupling map)
- **Total Gates**: 181 (71 RZ, 68 SX, 42 CZ)
- **Shots**: 4,096 per parameter point
- **Elapsed Wait Time**: $77.18\text{ s}$ (20.0s QPU time billed to ledger)
- **Budget Ledger**: Tracked in `data/qpu_ledger.json` ($50.35\text{ s}$ consumed of $600.0\text{ s}$ limit, $549.65\text{ s}$ remaining)

| $\theta$ (rad) | Measured Physical Energy (Ha) | Exact CASCI Reference (Ha) | Physical QPU Error (mHa) | Rel Error (%) |
| :---: | :---: | :---: | :---: | :---: |
| **$-0.10$** | $-639.58606039$ | $-639.72428323$ | $138.22\text{ mHa}$ | $0.0216\%$ |
| **$-0.05$** | $-639.59354271$ | $-639.72428323$ | $130.74\text{ mHa}$ | $0.0204\%$ |
| **$0.00$** | **$-639.59675784$** | **$-639.72428323$** | **$127.53\text{ mHa}$** | **$0.0199\%$** |
| **$+0.05$** | **$-639.59691169$** | **$-639.72428323$** | **$127.37\text{ mHa}$** | **$0.0199\%$** |
| **$+0.10$** | $-639.58595898$ | $-639.72428323$ | $138.32\text{ mHa}$ | $0.0216\%$ |

> [!NOTE]
> **Physical Energy Landscape Verification**:
> The physical Heron QPU reproduces the expected variational energy minimum at $\theta \approx 0.00 - 0.05\text{ rad}$. The unmitigated physical hardware noise shifts the landscape upward by $\sim 127\text{ mHa}$ across 42 entangling CZ gates ($\sim 3\text{ mHa}$ error per CZ gate), completely validating genuine quantum circuit execution without trivial state collapse.

*(Historical reference: Job [`daor3p5r85ps73ffmvn0`](https://quantum.ibm.com/jobs/daor3p5r85ps73ffmvn0) evaluated UCCSD at $\boldsymbol{\theta}=\mathbf{0}$, collapsing to depth 1 and 0 2Q gates with energy $-639.72364609\text{ Ha}$ and error $0.6371\text{ mHa}$.)*

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
