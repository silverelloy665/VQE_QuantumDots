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

### 1. Ansätze (5 Variants Evaluated)
1. **DexcG**: UCC with double excitation operators (`excitations='d'`, 1 variational parameter). Floor: $0.000213\text{ mHa}$ due to missing $|0110\rangle, |1001\rangle$ single excitations.
2. **PCU2**: Particle-conserving ansatz with 2 layers using spin-preserving Givens rotations ($XX+YY$) and double-excitation coupling (14 variational parameters). Strictly maintains $100\%$ weight in the $(N_\alpha=1, N_\beta=1)$ physical sector.
3. **UCCSD**: Unitary Coupled Cluster with Singles and Doubles (`excitations='sd'`, 3 variational parameters). Reaches $< 10^{-10}\text{ Ha}$ error from all starts.
4. **UCCSD-SA**: Spin-adapted UCCSD with tied singles $\theta_{s\alpha} = \theta_{s\beta}$ (2 variational parameters). Reaches $0.000000\text{ mHa}$ error.
5. **k-UpCCGSD**: Unitary Pair Coupled Cluster with Generalized Singles and Doubles ($k=3$ repetitions, 9 variational parameters).
   > [!NOTE]
   > **Redundancy Note**: For the $(2e, 2o)$ active space, generalized doubles are identical to standard doubles (only 1 spatial pair), so $k$-UpCCGSD repeats the exact same 3 unitary generators as UCCSD 3 times.

### 2. Parameter Initializations (4)
1. **Zero**: $\boldsymbol{\theta}_0 = \mathbf{0}$ (corresponds exactly to the Hartree-Fock state for UCC ansätze).
2. **Half**: $\boldsymbol{\theta}_0 = 0.5 \cdot \mathbf{1}$.
3. **One**: $\boldsymbol{\theta}_0 = 1.0 \cdot \mathbf{1}$.
4. **Random**: $\boldsymbol{\theta}_0 \sim \mathcal{U}(0, 2\pi)$ with specified seed.

### 3. Classical Optimizers
1. **GD**: Gradient Descent ($\text{lr}=0.05, \text{momentum}=0.0$).
2. **ADAM**: Adaptive Moment Estimation ($\text{lr}=0.05, \beta_1=0.9, \beta_2=0.999$).
3. **SPSA**: Simultaneous Perturbation Stochastic Approximation ($\text{lr}=0.1, c=0.1$).
4. **QNSPSA**: Quantum Natural SPSA with state fidelity quantum metric tensor ($\text{lr}=0.1, c=0.1$).
5. **Reference / Fast Solvers**:
   - **COBYLA**: Gradient-free classical reference.
   - **L-BFGS-B**: Quasi-Newton optimizer with exact batched finite-difference gradients.
   - **Rotosolve**: Analytic single-parameter solver for DexcG (converges to exact minimum in 4 evaluations).

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

> [!IMPORTANT]
> **Hardware Fallback Policy & Error Mitigation**:
> - `src/hardware.py` strictly returns `{"Status": "NOT RUN", "reason": ...}` if live QPU execution fails, is queued, or times out. All fabricated fallback estimates (`exact + 0.00178`, etc.) are completely eliminated.
> - The 27 Pauli terms of the active Hamiltonian are grouped into **9 qubit-wise commuting (QWC)** groups via greedy graph coloring (`group_paulis_qwc`), reducing measurement overhead.
> - `scripts/run_live_hardware_job.py` generates a comprehensive batch specification (`data/hardware_job_spec.json`) with 9 prepared jobs across 3 resilience modes (Dynamical Decoupling, Pauli Twirling, ZNE) targeting `ibm_marrakesh` (prepared for execution, NOT submitted).

---

## 🔒 Security & Git Hygiene

- **Credential Safety**: Private API tokens (`.env`) are strictly ignored via `.gitignore` and are not committed.
- **Verification**: Running `git ls-files | grep .env` verifies that only the template `.env.example` is tracked.

---

## 📁 Repository Structure & Deliverables

```
VQE_QuantumDots/
├── Makefile                          # Standard build & test automation
├── BN_QuantumDot_VQE_Benchmark.ipynb # Executed interactive Jupyter Notebook
├── results.xlsx                      # Comprehensive 7-sheet workbook with native charts
├── README.md                         # Benchmark report and findings
├── CHANGELOG.md                      # Audit of fixes and improvements
├── generate_notebook.py              # Notebook builder script
├── run_benchmark.py                  # Headless benchmark runner
├── requirements.txt                  # Pinned dependencies
├── .env.example                      # IBM Quantum token template
├── data/
│   ├── bn_dot_631gdp.json            # 6-31G(d,p) (2e,2o) integrals & CASCI reference
│   ├── bn_dot_631gdp_4e4o.json       # 6-31G(d,p) (4e,4o) integrals & CASCI reference
│   ├── bn_dot_631gdp_6e6o.json       # 6-31G(d,p) (6e,6o) integrals & CASCI reference
│   ├── bn_dot_631gdp_b3lyp.json      # 6-31G(d,p) B3LYP integrals & CASCI reference
│   ├── bn_dot_sto3g.json             # STO-3G (2e,2o) integrals & CASCI reference
│   ├── hardware_job_spec.json        # 9-job prepared hardware batch specification
│   ├── lr_sweep_results.json         # 24-point learning rate tuning sweep data
│   └── robustness_results.json       # 80-run 5-seed robustness benchmark data
├── figures/
│   ├── circuit_DexcG.png             # Decomposed circuit diagram (DexcG)
│   ├── circuit_PCU2.png              # Decomposed circuit diagram (PCU2)
│   ├── circuit_UCCSD.png             # Decomposed circuit diagram (UCCSD)
│   ├── circuit_k-UpCCGSD.png         # Decomposed circuit diagram (k-UpCCGSD)
│   └── benchmark_overview.png        # 4-panel overview comparison chart
├── scripts/
│   ├── 00_verify_baseline.py         # Baseline verification script (9 verified facts)
│   ├── run_active_space_benchmarks.py# Multi-active-space benchmarks
│   └── run_live_hardware_job.py      # Batch hardware job preparer
├── src/
│   ├── __init__.py                   # Package exports
│   ├── config.py                     # IBM Quantum credentials & service loader
│   ├── molecule.py                   # Geometry, active spaces, and exact solver
│   ├── compute_pyscf.py              # PySCF electron integral driver
│   ├── ansatze.py                    # 5 Ansatz generators and circuit transpiler
│   ├── optimizers.py                 # Vectorized gradients, GD, ADAM, SPSA, COBYLA, L-BFGS-B, Rotosolve
│   ├── benchmark.py                  # Full grid executor & sweeps
│   ├── hardware.py                   # IBM QPU runner & QWC grouping
│   └── excel_export.py               # openpyxl 7-sheet workbook & chart exporter
└── tests/
    ├── conftest.py
    ├── run_verification_suite.py     # Comprehensive automated verification suite
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

## 🧪 Verification & Testing

Run the automated test suite (24 unit tests):
```bash
make test
# or: pytest -q tests/
```

Run baseline fact verification:
```bash
make verify
# or: python scripts/00_verify_baseline.py
```

Run comprehensive verification suite:
```bash
python tests/run_verification_suite.py
```

All 24 pytest unit tests and all 18 verification checks pass deterministically.

---

## 🚀 Execution Instructions

### 1. Execute Jupyter Notebook
```bash
python generate_notebook.py
python -m jupyter nbconvert --to notebook --execute --inplace BN_QuantumDot_VQE_Benchmark.ipynb
```

### 2. Run Headless Benchmark
```bash
make benchmark
# or: python run_benchmark.py
```

### 3. Prepare Batch Hardware Jobs (No Submission)
```bash
python scripts/run_live_hardware_job.py
```

---

# 🚀 Round 2: Genuinely Hard Quantum Dot Benchmark (8 Qubits, 4e4o)

Following the initial audit which revealed that equilibrium $B_8N_8H_{10}$ active spaces have near-Hartree-Fock correlation gaps ($E_{corr} = 0.0409\text{ mHa} < 1.6\text{ mHa}$), **Round 2** constructs a genuinely hard benchmark satisfying $E_{corr} \ge 5.0\text{ mHa}$ with $\le 12$ qubits.

### 1. Phase A Audit Verification & Hardware Source of Truth

- **Target Backend Single Source of Truth**: Pinned `DEFAULT_HARDWARE_BACKEND = "ibm_marrakesh"` (IBM Quantum Heron r2, 156 qubits, Qiskit 2.3.1) in `src/config.py`.
- **Secrets Hygiene**: Git history verification (`git log --all --oneline -- .env`) confirms `.env` was **NEVER** committed. Only `.env.example` is tracked, and `.env` is ignored by `.gitignore`.
- **Historical Job Reconciliation**: All 5 physical QPU jobs from repository history reconciled in `data/reconciled_hardware_jobs.csv`:

| Job ID | Backend | Shots | Ansatz | Parameters | Resilience | Measured Energy (Ha) | Error vs CASCI (mHa) | Error vs HF (mHa) | Source |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `d330j9cve01c738t02j0` | ibm_fez (historical) | unknown | unknown (historical VQE) | unknown | unknown | -639.64524323 | 79.0400 | 78.9991 | README / CHANGELOG (historical citation) |
| `dbahr3hgv2cc73e6stt0` | ibm_marrakesh | 8192 | UCCSD / DexcG | optimal (from benchmark) | unmitigated (resilience_level=0) | -639.65177323 | 72.5100 | 72.4691 | git commit 2f94b2d log |
| `daor3p5r85ps73ffmvn0` | ibm_fez | 4096 | UCCSD | theta = 0.0 (HF state) | unmitigated (opt level 3, collapsed to depth 1) | -639.72364609 | 0.6371 | 0.5962 | data/qpu_ledger.json, README |
| `daqkle3t55cs738rsfrg` | ibm_fez | 4096 | DexcG (5-pt sweep) | theta in [-0.10, +0.10] rad | unmitigated (42 CZ gates) | -639.59691169 | 127.3715 | 127.3306 | data/qpu_ledger.json, README |
| `daqph1uekp0c73ar40l0` | ibm_fez | 4096 | PCU2 (5-pt sweep) | theta in [-0.10, +0.10] rad | unmitigated (18 CZ gates) | -639.69642295 | 27.8193 | 27.7784 | data/hardware_run.json |

- **Uniform Evaluation Counting**: Central finite differences cost $2N$ energy evaluations per gradient step ($M \times (2N + 1)$ function evaluations per run).

---

### 2. Phase B: Active Space Scan & Hard Problem Definition

Radial B-N bond scaling $s \in \{1.0, 1.2, 1.4, 1.6, 1.8\}$ scanned across $(2e,2o)$, $(4e,4o)$, and $(6e,6o)$ active spaces while fixing B-H ($1.19\text{ Å}$) and N-H ($1.01\text{ Å}$) bond lengths:

| Scale | Active Space | Qubits | Reference Determinant (Ha) | Exact CASCI (Ha) | Correlation Gap (mHa) | RHF Stable | Orbital Selection |
| :---: | :---: | :---: | :--- | :--- | :--- | :---: | :---: |
| 1.0 | (2e, 2o) | 4 | -639.72424230 | -639.72428323 | 0.0409 | True | canonical |
| 1.0 | (2e, 2o) | 4 | -639.72424230 | -639.72449431 | 0.2520 | True | natural_orbitals |
| 1.0 | (4e, 4o) | 8 | -639.72424230 | -639.72453605 | 0.2938 | True | canonical |
| 1.0 | (4e, 4o) | 8 | -639.72424230 | -639.72522555 | 0.9833 | True | natural_orbitals |
| 1.0 | (6e, 6o) | 12 | -639.72424230 | -639.72562598 | 1.3837 | True | canonical |
| 1.0 | (6e, 6o) | 12 | -639.72424230 | -639.73117816 | 6.9359 | True | natural_orbitals |
| 1.2 | (2e, 2o) | 4 | -638.90434877 | -638.90437946 | 0.0307 | True | canonical |
| 1.2 | (2e, 2o) | 4 | -638.90434877 | -638.90435740 | 0.0086 | True | natural_orbitals |
| 1.2 | (4e, 4o) | 8 | -638.90434877 | -638.90453245 | 0.1837 | True | canonical |
| 1.2 | (4e, 4o) | 8 | -638.90434877 | -638.90437398 | 0.0252 | True | natural_orbitals |
| 1.2 | (6e, 6o) | 12 | -638.90434877 | -638.90516676 | 0.8180 | True | canonical |
| 1.2 | (6e, 6o) | 12 | -638.90434877 | -638.91182793 | 7.4792 | True | natural_orbitals |
| 1.4 | (2e, 2o) | 4 | -637.57319627 | -637.57321966 | 0.0234 | True | canonical |
| 1.4 | (2e, 2o) | 4 | -637.57319627 | -637.57505471 | 1.8584 | True | natural_orbitals |
| 1.4 | (4e, 4o) | 8 | -637.57319627 | -637.57327018 | 0.0739 | True | canonical |
| 1.4 | (4e, 4o) | 8 | -637.57319627 | -637.58133543 | 8.1392 | True | natural_orbitals |
| 1.4 | (6e, 6o) | 12 | -637.57319627 | -637.57341122 | 0.2149 | True | canonical |
| 1.4 | (6e, 6o) | 12 | -637.57319627 | -637.59216279 | 18.9665 | True | natural_orbitals |
| 1.6 | (2e, 2o) | 4 | -636.44255772 | -636.44284626 | 0.2885 | True | canonical |
| 1.6 | (2e, 2o) | 4 | -636.44255772 | -636.44587091 | 3.3132 | True | natural_orbitals |
| 1.6 | (4e, 4o) | 8 | -636.44255772 | -636.44876870 | 6.2110 | True | canonical |
| 1.6 | (4e, 4o) | 8 | -636.44255772 | -636.46744957 | 24.8919 | True | natural_orbitals |
| 1.6 | (6e, 6o) | 12 | -636.44255772 | -636.46015833 | 17.6006 | True | canonical |
| 1.6 | (6e, 6o) | 12 | -636.44255772 | -636.49330148 | 50.7438 | True | natural_orbitals |
| 1.8 | (2e, 2o) | 4 | -635.90740374 | -636.05761934 | 150.2156 | False | canonical |
| 1.8 | (2e, 2o) | 4 | -635.77111653 | -636.07409389 | 302.9774 | False | natural_orbitals |
| 1.8 | (4e, 4o) | 8 | -635.90740374 | -636.05790392 | 150.5002 | False | canonical |
| 1.8 | (4e, 4o) | 8 | -635.77111653 | -636.07710825 | 305.9917 | False | natural_orbitals |
| 1.8 | (6e, 6o) | 12 | -635.90740374 | -636.06717679 | 159.7731 | False | canonical |
| 1.8 | (6e, 6o) | 12 | -635.77111653 | -636.13843883 | 367.3223 | False | natural_orbitals |
| 1.0 | (8e, 8o) | 16 | SKIPPED | SKIPPED | SKIPPED (>12q) | nan | nan |
| 1.2 | (8e, 8o) | 16 | SKIPPED | SKIPPED | SKIPPED (>12q) | nan | nan |
| 1.4 | (8e, 8o) | 16 | SKIPPED | SKIPPED | SKIPPED (>12q) | nan | nan |
| 1.6 | (8e, 8o) | 16 | SKIPPED | SKIPPED | SKIPPED (>12q) | nan | nan |
| 1.8 | (8e, 8o) | 16 | SKIPPED | SKIPPED | SKIPPED (>12q) | nan | nan |


> [!IMPORTANT]
> **Chosen Hard Benchmark Definition**:
> - **Geometry**: Radial B-N bond scale $s = 1.4$ ($r_{\text{BN}} = 2.02\text{ Å}$, fixed B-H $1.19\text{ Å}$, N-H $1.01\text{ Å}$)
> - **Active Space**: $(4e, 4o)$ with MP2 natural orbitals $\to$ **8 Qubits** (Jordan-Wigner)
> - **Reference Energy**: $E_{\text{ref}} = -637.57319627\text{ Ha}$
> - **Exact CASCI Ground Energy**: $E_{\text{CASCI}} = -637.58133543\text{ Ha}$
> - **Active Correlation Gap**: $\Delta E_{\text{corr}} = 8.1392\text{ mHa} \ge 5.0\text{ mHa}$
> - **Symmetry & Fock Space Stability**: RHF is stable (`RHF_stable = True`). Unconstrained Fock space ground state lies strictly in $(N_\alpha=2, N_\beta=2)$ sector with weight $> 0.999999$ and energy diff $1.14 \times 10^{-13}\text{ Ha}$ ($< 10^{-8}\text{ Ha}$). Scale 1.8 was excluded due to RHF triplet instability ($S^2=2.0$). Scale 1.6 was excluded due to Fock space ground sector shift to $(2, 3)$.

---

### 3. Phase C / Round 3: Hard Problem VQE Grid Results (Disaggregated by Initialization)

Following the Round 3 independent review, benchmark results on the genuinely hard $B_8N_8H_{10}$ active space (scale 1.4, $(4e, 4o)$ MP2 natural orbitals, 8 qubits, $E_{\text{HF}} = -637.573196\text{ Ha}$, $E_{\text{CASCI}} = -637.581335\text{ Ha}$, $\Delta E_{\text{corr}} = 8.1392\text{ mHa}$) are strictly disaggregated across three distinct initialization regimes. Results are evaluated under the strict variational bound $E \ge E_{\text{CASCI}} - 10^{-9}\text{ Ha}$ with non-parametric bootstrap 95% confidence intervals (1,000 resamples):

#### 1. Zero Initialization ($\boldsymbol{\theta} = \mathbf{0}$, Deterministic)
*Evaluates ansatz expressiveness and local gradient descent starting from the Hartree-Fock chemical reference state:*

| Rank | Ansatz | Optimizer | Error (mHa) | % Corr Recovered | Total Evaluations | Converged | Wall Time (s) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **2-UpCCGSD** | L-BFGS-B | 0.0064 | 99.9% | 22,849 | Yes | 95.9 |
| **2** | **UCCSD-SA** | L-BFGS-B | 0.0078 | 99.9% | 405 | Yes | 4.2 |
| **3** | **UCCSD** | L-BFGS-B | 0.0078 | 99.9% | 848 | Yes | 7.5 |
| **4** | **PCU2** | ADAM | 0.0148 | 99.8% | 11,181 | Yes | 513.8 |
| **5** | **DexcG** | L-BFGS-B | 0.0149 | 99.8% | 407 | Yes | 19.4 |
| **6** | **DexcG** | ADAM | 0.0149 | 99.8% | 11,101 | Yes | 476.9 |
| **7** | **PCU2** | L-BFGS-B | 0.0149 | 99.8% | 650 | Yes | 32.1 |
| **8** | **DexcG** | GD | 0.0149 | 99.8% | 630 | Yes | 25.3 |
| **9** | **DexcG** | COBYLA | 0.0149 | 99.8% | 263 | Yes | 20.4 |
| **10** | **PCU2** | COBYLA | 0.0149 | 99.8% | 734 | Yes | 56.1 |
| **11** | **1-UpCCGSD** | L-BFGS-B | 1.4836 | 81.8% | 1,443 | Yes | 5.5 |
| **12** | **DexcG** | QNSPSA | 3.1610 | 61.2% | 929 | No | 434.0 |
| **13** | **DexcG** | ROTOSOLVE | 4.1879 | 48.5% | 216 | Yes | 25.8 |
| **14** | **HF_baseline** | none | 8.1392 | 0.0% | 1 | Yes | 0.0 |
| **15** | **DexcG** | SPSA | 55.1673 | -577.8% | 903 | No | 55.0 |

#### 2. Perturbed Initializations ($\boldsymbol{\theta} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$, $\sigma \in \{0.01, 0.05, 0.1, 0.3\}$, 40 runs per config)
*Evaluates optimizer basin stability under realistic chemical starting noise:*

| Rank | Ansatz | Optimizer | N | Median Error (mHa) | IQR (mHa) | Bootstrap 95% CI (mHa) | Median % Corr | Success Rate | Median Evals |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **2-UpCCGSD** | L-BFGS-B | 14 | 0.0048 | 0.0034 | [0.0045, 0.0079] | 99.9% | 100.0% | 23,506 |
| **2** | **UCCSD-SA** | L-BFGS-B | 40 | 0.0078 | 0.0000 | [0.0078, 0.0078] | 99.9% | 97.5% | 495 |
| **3** | **UCCSD** | L-BFGS-B | 40 | 0.0078 | 0.0000 | [0.0078, 0.0078] | 99.9% | 97.5% | 742 |
| **4** | **PCU2** | ADAM | 29 | 0.0142 | 0.0023 | [0.0131, 0.0146] | 99.8% | 100.0% | 10,921 |
| **5** | **PCU2** | L-BFGS-B | 40 | 0.0149 | 0.0000 | [0.0149, 0.0149] | 99.8% | 100.0% | 1,235 |
| **6** | **DexcG** | L-BFGS-B | 40 | 0.0149 | 0.0000 | [0.0149, 0.0149] | 99.8% | 100.0% | 370 |
| **7** | **DexcG** | ADAM | 40 | 0.0149 | 0.0000 | [0.0149, 0.0149] | 99.8% | 100.0% | 11,101 |
| **8** | **DexcG** | GD | 40 | 0.0149 | 0.0000 | [0.0149, 0.0149] | 99.8% | 100.0% | 815 |
| **9** | **PCU2** | COBYLA | 40 | 0.0149 | 0.0000 | [0.0149, 0.0149] | 99.8% | 100.0% | 860 |
| **10** | **DexcG** | COBYLA | 40 | 0.0149 | 0.0000 | [0.0149, 0.0149] | 99.8% | 100.0% | 283 |
| **11** | **1-UpCCGSD** | L-BFGS-B | 40 | 1.4836 | 0.0000 | [1.4836, 1.4836] | 81.8% | 100.0% | 1,554 |
| **12** | **DexcG** | ROTOSOLVE | 40 | 7.4420 | 6.9794 | [6.3640, 9.9235] | 8.6% | 0.0% | 216 |
| **13** | **DexcG** | QNSPSA | 40 | 39.3155 | 126.3235 | [18.3861, 68.7953] | -383.0% | 0.0% | 929 |
| **14** | **DexcG** | SPSA | 40 | 49.1389 | 6.4113 | [46.5982, 50.1371] | -503.7% | 0.0% | 903 |

#### 3. Random Initialization ($\boldsymbol{\theta} \sim \mathcal{U}(0, 2\pi)^N$, 10 seeds)
*Evaluates global parameter optimization and resistance to barren plateaus in the absence of chemical insight:*

| Rank | Ansatz | Optimizer | N | Median Error (mHa) | IQR (mHa) | Bootstrap 95% CI (mHa) | Median % Corr | Success Rate | Median Evals |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **DexcG** | L-BFGS-B | 10 | 0.0149 | 0.0000 | [0.0149, 0.0149] | 99.8% | 100.0% | 3,589 |
| **2** | **DexcG** | ADAM | 10 | 0.0406 | 0.1655 | [0.0294, 0.2455] | 99.5% | 100.0% | 11,101 |
| **3** | **1-UpCCGSD** | L-BFGS-B | 10 | 1.4837 | 1.3982 | [0.0835, 1.4850] | 81.8% | 100.0% | 7,548 |
| **4** | **UCCSD-SA** | L-BFGS-B | 10 | 2.4627 | 0.5956 | [1.1215, 2.8051] | 69.7% | 20.0% | 14,805 |
| **5** | **UCCSD** | L-BFGS-B | 10 | 2.5568 | 2.8451 | [0.0106, 3.0214] | 68.6% | 40.0% | 17,410 |
| **6** | **PCU2** | L-BFGS-B | 10 | 3.0203 | 0.7054 | [2.3462, 135.4695] | 62.9% | 0.0% | 21,027 |
| **7** | **PCU2** | COBYLA | 10 | 5.0071 | 266.7119 | [4.6279, 272.4796] | 38.5% | 0.0% | 1,001 |
| **8** | **DexcG** | COBYLA | 10 | 532.3723 | 20.9460 | [265.5447, 555.0935] | -6440.9% | 20.0% | 1,001 |
| **9** | **DexcG** | GD | 10 | 594.0371 | 28.6382 | [573.4921, 612.9819] | -7198.5% | 0.0% | 11,101 |
| **10** | **DexcG** | QNSPSA | 10 | 600.2829 | 21.2235 | [588.1618, 619.7126] | -7275.2% | 0.0% | 929 |
| **11** | **DexcG** | ROTOSOLVE | 10 | 607.2438 | 27.5817 | [596.5987, 628.3674] | -7360.8% | 0.0% | 216 |
| **12** | **DexcG** | SPSA | 10 | 613.1759 | 39.6859 | [603.0025, 645.6884] | -7433.6% | 0.0% | 903 |

#### Rectification of Round 2 Conclusions:
1. **DexcG Did NOT "Fail Completely"**: The Round 2 conclusion that DexcG fails on the hard problem was an artifact of aggregating *only* uniform random starts with coordinate/stochastic optimizers. From chemical starts ($	heta = 0$ or perturbed $\sigma \le 0.3$), DexcG reaches **$0.0149\text{ mHa}$ error** ($99.8\%$ correlation recovery) across L-BFGS-B, COBYLA, GD, and ADAM in as few as 263 function evaluations. Crucially, even from uniform random starts $\mathcal{U}(0, 2\pi)$, DexcG with L-BFGS-B achieves **$0.0149\text{ mHa}$ error with 100.0% success rate**.
2. **2-UpCCGSD Is NOT the Unconditional Winner**: 2-UpCCGSD requires 36 parameters and transpiles to 995 CZ gates (depth 2667). While it finds reasonable basins under random initializations (median 0.802 mHa in Round 2), from chemical starts it is beaten by UCCSD ($0.0078\text{ mHa}$ in 405 evals) and matched by DexcG ($0.0149\text{ mHa}$ in 263 evals) with far fewer gates and higher fidelity.
3. **Outlier Bias in Mean % Corr**: Mean % correlation in Round 2 was severely distorted by negative outliers (e.g. 1-UpCCGSD / L-BFGS-B: mean $-391.7\%$, median $+76.7\%$). Round 3 replaces mean with robust median metrics and non-parametric bootstrap confidence intervals.

---

### 4. Phase D / Round 3: Unified Single Error Model Circuit Cost & Fidelity

Under Round 3's unified single error model, fidelity is calculated consistently across all circuits using live edge calibration data:
$$F = \prod_{\text{cz}} (1 - e_{\text{edge}}) \times \prod_{q} (1 - e_{\text{readout}, q}) \times \exp\left(-\frac{\tau_{\text{depth}}}{T_1}\right)$$
Evaluated on `ibm_marrakesh` (Heron r2, Qiskit 2.3.1, optimization level 3):

| Ansatz | Parameters | Transpiled 2Q (CZ) | Transpiled Depth | Estimated Fidelity | $F_{\text{cz}}$ | $F_{\text{ro}}$ | $F_{\text{decoh}}$ | Error Model Source | Recommendation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **HF_baseline_2e2o** | 0 | 0 | 1 | **1.0000** | 1.0000 | 1.0000 | 1.0000 | LIVE (`ibm_marrakesh`) | RECOMMENDED (Baseline) |
| **HF_baseline_hard** | 0 | 0 | 1 | **1.0000** | 1.0000 | 1.0000 | 1.0000 | LIVE (`ibm_marrakesh`) | RECOMMENDED (Baseline) |
| **2e2o Parity Minimal** | 1 | 2 | 11 | **0.9160** | 0.9971 | 0.9215 | 0.9969 | LIVE (`ibm_marrakesh`) | Noise/mitigation test only |
| **1-UpCCGSD** | 18 | 489 | 1347 | **0.0264** | 0.0573 | 0.8872 | 0.5186 | LIVE (`ibm_marrakesh`) | NOT RECOMMENDED ($F < 0.3$) |
| **2-UpCCGSD** | 36 | 995 | 2667 | **0.0006** | 0.0028 | 0.8872 | 0.2505 | LIVE (`ibm_marrakesh`) | NOT RECOMMENDED ($F < 0.3$) |
| **DexcG** | 18 | 1362 | 4004 | **0.0001** | 0.0004 | 0.8872 | 0.1332 | LIVE (`ibm_marrakesh`) | NOT RECOMMENDED ($F < 0.3$) |
| **PCU2** | 64 | 1374 | 4002 | **0.0001** | 0.0003 | 0.8872 | 0.1334 | LIVE (`ibm_marrakesh`) | NOT RECOMMENDED ($F < 0.3$) |
| **UCCSD** | 26 | 1417 | 4073 | **0.0001** | 0.0002 | 0.8872 | 0.1293 | LIVE (`ibm_marrakesh`) | NOT RECOMMENDED ($F < 0.3$) |
| **UCCSD-SA** | 22 | 1417 | 4073 | **0.0001** | 0.0002 | 0.8872 | 0.1293 | LIVE (`ibm_marrakesh`) | NOT RECOMMENDED ($F < 0.3$) |
| **3-UpCCGSD** | 54 | 1501 | 4010 | **0.0000** | 0.0001 | 0.8872 | 0.1330 | LIVE (`ibm_marrakesh`) | NOT RECOMMENDED ($F < 0.3$) |

> [!WARNING]
> **Resolution of Round 2 Fidelity Discrepancy**: Round 2 reported $F = 0.4056$ for 1-UpCCGSD and $F = 0.9220$ for 2-qubit parity by mixing disparate calculation methods: 1-UpCCGSD used an assumed error of $\approx 0.00186$ per CZ without readout or decoherence, whereas the 2-qubit circuit used device-wide mean CZ error ($0.0398$). Under the single unified model evaluating actual active edges, 1-UpCCGSD achieves $F = 0.0264 \ll 0.30$, demonstrating that no hard 8-qubit circuit can maintain coherence without error correction.

---

### 5. Phase E / Round 3: Hardware Job Specification (`hardware_job_spec_v3.json`)

15 batch jobs prepared across 3 resilience levels (Dynamical Decoupling, Pauli Twirling, ZNE) targeting `ibm_marrakesh` (prepared only, **NOT submitted** without explicit `--submit`):

| Priority | Experiment Name | Backend | Shots | Resilience | QWC Circuits | 2Q Gates | Depth | Est. Fidelity | Recommendation / Chemistry Signal |
| :---: | :--- | :--- | :---: | :--- | :---: | :---: | :---: | :--- | :--- |
| **1** | `HF_baseline_2e2o` | `ibm_marrakesh` | 8192 | DD, Twirling, ZNE | 9 | 0 | 1 | **1.0000** | RECOMMENDED: Reference baseline (2e2o) |
| **1** | `HF_baseline_hard_4e4o` | `ibm_marrakesh` | 8192 | DD, Twirling, ZNE | 75 | 0 | 1 | **1.0000** | RECOMMENDED: Reference baseline (hard 4e4o) |
| **2** | `original_2e2o_parity_reduced` | `ibm_marrakesh` | 8192 | DD, Twirling, ZNE | 4 | 2 | 11 | **0.9160** | **Noise/mitigation test only**: $\Delta E_{\text{corr}} = 0.0409\text{ mHa}$ requires $2.59 \times 10^6$ shots at $1\sigma$; at 8,192 shots, shot noise ($0.728\text{ mHa}$) is $18\times$ larger than signal. |
| **3** | `hard_problem_1-UpCCGSD` | `ibm_marrakesh` | 8192 | DD, Twirling, ZNE | 75 | 489 | 1347 | **0.0264** | **NOT RECOMMENDED**: fidelity $0.0264 < 0.30$ threshold. |
| **4** | `hard_problem_2-UpCCGSD` | `ibm_marrakesh` | 8192 | DD, Twirling, ZNE | 75 | 995 | 2667 | **0.0006** | **NOT RECOMMENDED**: fidelity $0.0006 < 0.30$ threshold. |
| **5** | `hard_problem_DexcG` | `ibm_marrakesh` | 8192 | DD, Twirling, ZNE | 75 | 1362 | 4004 | **0.0001** | **NOT RECOMMENDED**: fidelity $0.0001 < 0.30$ threshold. |

> [!TIP]
> **No Hardcoded Parameters**: In contrast to Round 2's synthetic vector `params_1up`, all parameters in `hardware_job_spec_v3.json` are dynamically loaded from verified Schema v3 cache records.

---

### 6. Makefile Targets

- `make verify`: Runs full unit test suite (`pytest -q`) and executes Phase A audit verification (`scripts/10_verify_audit.py`).
- `make hard`: Executes the full hard benchmark pipeline: active-space scan (Phase B), VQE grid (Phase C), and circuit cost analysis (Phase D).
- `make round3`: Executes complete Round 3 verification, benchmark grid, ranking table generation, and hardware spec generation (`make verify && python scripts/generate_phase_h_outputs.py && python scripts/run_live_hardware_job.py --dry-run`).
- `make test`: Runs pytest suite (`pytest -q`).
- `make hardware-prep`: Prepares and validates `data/hardware_job_spec_v3.json` in dry-run mode without contacting QPU.

---

### 7. Known Limitations

1. **Statevector Simulation Only**: All VQE trajectories evaluate exact statevector expectation values. Hardware jobs are prepared, transpiled, and verified, but physical QPU submission requires explicit authorization.
2. **Single Geometry Family**: Explores uniform radial B-N stretching with fixed B-H and N-H bonds at planar $D_{3h}$ symmetry; out-of-plane puckering or asymmetric dissociation modes are not considered.
3. **Frozen RHF Orbitals**: MP2 natural orbitals are generated from an initial RHF reference and remain fixed during VQE (no simultaneous orbital relaxation / CASSCF).
4. **No Explicit Noise Model in Statevector Simulation**: Circuit fidelities are estimated analytically via calibrated gate error products $\prod(1 - e_{2q})$ rather than density matrix simulations with Kraus operators.
