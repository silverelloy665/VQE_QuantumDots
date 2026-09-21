# CHANGELOG: VQE Quantum Dots Benchmark Overhaul

All notable changes, bug fixes, and mathematical corrections made to the `VQE_QuantumDots` repository are documented in this changelog.

---

## [2.0.0] - Quantum Benchmark Overhaul & Mathematical Corrections

### 1. Corrected Reference Electronic Energies & Active Space Gaps
- **Issue**: The previous documentation reported incorrect RHF energies (`-639.68334407 Ha` for 6-31G(d,p) and `-631.70087799 Ha` for STO-3G), fabricating an artificial "40.9 mHa" gap to the CASCI ground state.
- **Correction**: Recomputed the exact full-electron RHF and CASCI $(2e, 2o)$ active space electronic energies in PySCF 2.14.0 across multiple basis sets and active space sizes:
  - **STO-3G $(2e, 2o)$**: RHF = `-631.74167448 Ha`, Exact CASCI = `-631.74178346 Ha` ($E_{\text{corr}} = 0.10897\text{ mHa}$).
  - **6-31G(d,p) $(2e, 2o)$**: RHF = `-639.72424230 Ha`, Exact CASCI = `-639.72428323 Ha` ($E_{\text{corr}} = 0.04093\text{ mHa}$).
  - **Active Space Scaling**: Added $(4e, 4o)$ ($E_{\text{corr}} = 0.29375\text{ mHa}$), $(6e, 6o)$ ($E_{\text{corr}} = 1.38368\text{ mHa}$), and B3LYP orbital benchmarks.
  - Added warning banner in `src/molecule.py` and documentation clarifying that $(2e, 2o)$ is predominantly closed-shell, explaining why zero-initialization starts within $0.041\text{ mHa}$ of exact CASCI.

| Parameter | Previous (Fabricated) | Corrected (Computed) |
| :--- | :--- | :--- |
| **6-31G(d,p) RHF Energy** | `-639.68334407 Ha` | **`-639.72424230 Ha`** |
| **6-31G(d,p) CASCI Ground State** | `-639.72428323 Ha` | **`-639.72428323 Ha`** |
| **Active Correlation Gap $E_{\text{corr}}$** | `40.9 mHa` (Fabricated) | **`0.04093 mHa`** (Exact) |
| **STO-3G RHF Energy** | `-631.70087799 Ha` | **`-631.74167448 Ha`** |
| **STO-3G CASCI Ground State** | `-631.74178346 Ha` | **`-631.74178346 Ha`** |

---

### 2. Fixed Invalid Parameter-Shift Rule for UCC Operators
- **Issue**: `compute_gradient_batched` used a $\pi/2$ parameter-shift rule. However, UCC excitation operators $e^{\theta(T - T^\dagger)}$ mapped under Jordan-Wigner transformation are $\pi$-periodic in $\theta$. Evaluating $\frac{E(\theta+\pi/2) - E(\theta-\pi/2)}{2\sin(\pi/2)}$ resulted in a gradient of exactly `0.000000` at all tested non-zero points, causing Gradient Descent to fail.
- **Correction**: Replaced with vectorized **central finite differences** ($\epsilon = 10^{-5}$) in a single batched PUB call evaluating all $2N$ points $(E(\boldsymbol{\theta} + \epsilon \mathbf{e}_i), E(\boldsymbol{\theta} - \epsilon \mathbf{e}_i))$ simultaneously.
- **Verification**: Added `tests/test_gradients.py` verifying agreement with independent single-point finite differences ($< 10^{-4}$) and verifying strictly monotonic energy descent for Gradient Descent on DexcG.

---

### 3. Corrected Circuit Structural Metrics & Parameter Counts
- **Issue**: README and documentation had mismatched parameter counts and circuit depths (e.g. PCU2 claimed 16 parameters; DexcG claimed depth 136; k-UpCCGSD claimed 12 parameters).
- **Correction**: Recomputed exact parameter counts and transpilation metrics with Qiskit 2.3.1 pass managers on `GenericBackendV2(5)`:
  - **DexcG**: 1 parameter, 8 raw 2Q gates, transpiled depth = 111, transpiled 2Q gates = 42.
  - **PCU2**: 14 parameters, 6 raw 2Q gates, transpiled depth = 65, transpiled 2Q gates = 18.
  - **UCCSD**: 3 parameters, 10 raw 2Q gates, transpiled depth = 124, transpiled 2Q gates = 49.
  - **k-UpCCGSD** ($k=3$): 9 parameters, 30 raw 2Q gates, transpiled depth = 372, transpiled 2Q gates = 145.

---

### 4. Purged Fabricated Hardware Energies & Mock Fallbacks
- **Issue**: `src/hardware.py` contained hardcoded fake energy offsets (`+0.00185`, `+0.00045`) and mock job IDs.
- **Correction**:
  - Removed all mock fallback values and fake energy offsets.
  - Added calibrated noisy Aer simulation using `FakeFez` noise model (4096 shots).
  - Explicitly marked historical job `d330j9cve01c738t02j0` as `UNVERIFIED - confirm in IBM Quantum dashboard`.
  - Added parameter-binding transpilation analysis demonstrating how binding $\boldsymbol{\theta}=\mathbf{0}$ before compilation eliminates all entangling gates (collapsing to the reference state).

---

### 5. Multi-Tier Tie Ranking & Performance Reporting
- **Issue**: Multiple configurations achieved identical final energies, but rankings did not break ties deterministically or report function evaluation budgets.
- **Correction**:
  - Implemented explicit multi-tier tie ranking:
    1. **Primary**: `Error_mHa` (lowest energy error)
    2. **Secondary**: `Total_Evaluations` (fewest function evaluations)
    3. **Tertiary**: `Wall_Time_s` (shortest execution time)
  - Added `% Correlation Recovered` metric:
    $$\% \text{Corr} = \frac{E_{\text{HF}} - E_{\text{VQE}}}{E_{\text{HF}} - E_{\text{exact}}} \times 100\%$$
  - Added 24-point learning rate sweep (`data/lr_sweep_results.json`) and 80-run 5-seed random robustness benchmark (`data/robustness_results.json`).

---

### 6. Excel Workbook Overhaul (`results.xlsx`)
- **Correction**: Expanded `results.xlsx` to **7 styled sheets** using `openpyxl` with 4 native charts:
  1. `Config`: Molecular geometry, active space parameters, environment versions, and LR sweep table.
  2. `Results`: Full 64-run benchmark grid with % correlation recovered, total evaluations, and 3-color conditional formatting.
  3. `Convergence`: 51 points ($t=0 \dots 50$) $\times$ 64 columns energy histories.
  4. `Summary`: Best per ansatz, tie-ranked top configurations, optimizer overview, and 4 native Excel charts.
  5. `Robustness`: 5-seed random initialization statistics.
  6. `Hardware`: Calibrated FakeFez noisy Aer simulation metrics and parameter binding test.
  7. `Verification`: Automated test suite PASS/FAIL results table.

---

### 7. Comprehensive 8-Module Verification Suite
- **Correction**: Added 8 automated test modules in `tests/` (15 unit tests) covering:
  - `test_ansatze.py`: Parameter counts and zero-parameter HF energy reproduction.
  - `test_b3lyp_active_space.py`: DFT orbital active space integrals and VQE.
  - `test_conservation.py`: Particle number ($\hat{N}$) and total spin ($\hat{S}_z$) expectation value conservation on optimized statevectors.
  - `test_geometry.py`: 26-atom geometry coordinates and B-N/B-H/N-H bond lengths.
  - `test_gradients.py`: Vectorized finite-difference accuracy and monotonic GD descent.
  - `test_hamiltonian.py`: Hamiltonian Jordan-Wigner decomposition and exact matrix diagonalization.
  - `test_pyscf_reference.py`: Reference RHF and CASCI active space energies.
  - `test_reproducibility.py`: Deterministic seed reproducibility across independent runs.
