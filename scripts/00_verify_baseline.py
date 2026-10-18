"""
scripts/00_verify_baseline.py
Reproduce VERIFIED FACTS 1-9 numerically and print as a table.
Run from repo root: python scripts/00_verify_baseline.py
"""
import sys
import warnings
import numpy as np
from pathlib import Path
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import Statevector

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from src.molecule import load_or_build_bn_dot_hamiltonian
from src.ansatze import get_ansatz_dict
from src.optimizers import compute_gradient_batched, get_initial_point

SEP = "=" * 72
PASS = "  [PASS]"
FAIL = "  [FAIL]"
WARN = "  [WARN]"


def fmt(label, measured, expected=None, unit="", atol=None):
    if expected is None:
        return f"  {label:50s}: {measured}"
    diff = abs(measured - expected)
    ok = (diff < atol) if atol is not None else True
    flag = PASS if ok else FAIL
    return f"  {label:50s}: {measured:.10f} {unit}  (expected {expected:.10f}, diff {diff:.2e}){flag}"


print(SEP)
print("  BASELINE VERIFICATION — VQE_QuantumDots / B8N8H10")
print(SEP)

# ── FACT 1: Active-space correlation energies ─────────────────────────────────
print("\n── FACT 1: Active-space energies ──────────────────────────────────────")

H_sto, E_sto_exact, meta_sto = load_or_build_bn_dot_hamiltonian(basis="sto-3g")
H_prod, E_prod_exact, meta_prod = load_or_build_bn_dot_hamiltonian(basis="6-31g(d,p)")

hf_sto    = meta_sto["hf_energy"]
hf_prod   = meta_prod["hf_energy"]
casci_sto = E_sto_exact
casci_prod = E_prod_exact

ecorr_sto  = abs(hf_sto  - casci_sto)  * 1000.0
ecorr_prod = abs(hf_prod - casci_prod) * 1000.0

print(fmt("  STO-3G  HF energy (Ha)",    hf_sto,    -631.74167448, "Ha", atol=1e-6))
print(fmt("  STO-3G  CASCI energy (Ha)", casci_sto, -631.74178346, "Ha", atol=1e-6))
print(fmt("  STO-3G  E_corr (mHa)",      ecorr_sto,   0.10898,     "mHa", atol=1e-3))
print(fmt("  6-31G   HF energy (Ha)",    hf_prod,   -639.72424230, "Ha", atol=1e-6))
print(fmt("  6-31G   CASCI energy (Ha)", casci_prod,-639.72428323, "Ha", atol=1e-6))
print(fmt("  6-31G   E_corr (mHa)",      ecorr_prod,  0.04093,     "mHa", atol=1e-3))

# ── FACT 2: theta=0 gives HF energy ─────────────────────────────────────────
print("\n── FACT 2: theta=0 → HF energy for all ansatze ──────────────────────")
ansatze = get_ansatz_dict()
estimator = StatevectorEstimator()
for name, qc in ansatze.items():
    n = qc.num_parameters
    theta0 = np.zeros(n)
    e0 = float(estimator.run([(qc, H_prod, [theta0])]).result()[0].data.evs[0])
    err_mHa = abs(e0 - hf_prod) * 1000.0
    tag = PASS if err_mHa < 0.1 else FAIL
    print(f"  {name:12s}  E(theta=0) = {e0:.8f} Ha  err_from_HF = {err_mHa:.6f} mHa{tag}")

# ── FACT 3: DexcG gradient at theta=0 with pi/2 shift ───────────────────────
print("\n── FACT 3: DexcG gradient at theta=0 ──────────────────────────────────")
qc_d = ansatze["DexcG"]
theta0 = np.zeros(qc_d.num_parameters)

# pi/2 shift (old broken method, implemented inline)
shift_half = np.pi / 2.0
p_plus  = theta0.copy(); p_plus[0]  += shift_half
p_minus = theta0.copy(); p_minus[0] -= shift_half
e_plus  = float(estimator.run([(qc_d, H_prod, [p_plus ])]).result()[0].data.evs[0])
e_minus = float(estimator.run([(qc_d, H_prod, [p_minus])]).result()[0].data.evs[0])
grad_pi2 = (e_plus - e_minus) / (2.0 * np.sin(shift_half))

# finite diff (truth)
eps = 1e-5
g_fd = compute_gradient_batched(qc_d, H_prod, theta0, estimator, method="finite_diff", eps=eps)
grad_true = float(g_fd[0])

print(f"  DexcG grad[0] at theta=0 via pi/2 param-shift : {grad_pi2:+.8f}  (should be ~0.0){PASS if abs(grad_pi2) < 1e-10 else FAIL}")
print(f"  DexcG grad[0] at theta=0 via finite diff       : {grad_true:+.8f}  (expected ~0.012613)")
tag = PASS if abs(abs(grad_true) - 0.012613) < 1e-4 else FAIL
print(f"  |grad_true - 0.012613| = {abs(abs(grad_true)-0.012613):.2e}{tag}")

# ── FACT 4: PCU2 particle non-conservation ───────────────────────────────────
print("\n── FACT 4: PCU2 particle (non-)conservation at random theta ───────────")
qc_pcu2 = ansatze["PCU2"]
n_pcu2 = qc_pcu2.num_parameters
print(f"  PCU2 num_parameters = {n_pcu2}  (spec says 14){PASS if n_pcu2==14 else FAIL}")

rng = np.random.RandomState(42)
theta_rand = rng.uniform(0, 1, n_pcu2)
sv = Statevector(qc_pcu2.assign_parameters(theta_rand))
probs = sv.probabilities_dict()

# compute weight in each particle sector
sector_weight = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
for bitstring, p in probs.items():
    n_ones = bitstring.count("1")
    sector_weight[n_ones] = sector_weight.get(n_ones, 0.0) + p

w2 = sector_weight.get(2, 0.0)
w_other = 1.0 - w2
print(f"  Weight in N=2 sector  : {w2:.6f}  (should be 1.0 for particle-conserving ansatz)")
print(f"  Weight in N≠2 sectors : {w_other:.6f}  (expect ~0.98 == NOT conserving){PASS if w_other > 0.5 else WARN}")

# gradients at theta=0 for PCU2
theta0_pcu2 = np.zeros(n_pcu2)
g_pcu2 = compute_gradient_batched(qc_pcu2, H_prod, theta0_pcu2, estimator, method="finite_diff", eps=eps)
all_zero = np.all(np.abs(g_pcu2) < 1e-8)
print(f"  All PCU2 gradients at theta=0 are zero: {all_zero}  max|g|={np.max(np.abs(g_pcu2)):.2e}{PASS if all_zero else FAIL}")

# ── FACT 5: DexcG floor ──────────────────────────────────────────────────────
print("\n── FACT 5: DexcG energy floor ─────────────────────────────────────────")
from scipy.optimize import minimize
def cost_dexcg(t):
    return float(estimator.run([(qc_d, H_prod, [t])]).result()[0].data.evs[0])

best = 1e10
for t0 in [np.array([0.0]), np.array([0.1]), np.array([-0.1]), np.array([np.pi/4])]:
    r = minimize(cost_dexcg, t0, method="L-BFGS-B", options={"maxiter": 500, "ftol": 1e-14})
    if r.fun < best:
        best = r.fun
floor_mHa = abs(best - casci_prod) * 1000.0
print(f"  DexcG floor energy    : {best:.10f} Ha")
print(f"  DexcG floor error     : {floor_mHa:.6f} mHa  (spec: ~0.000213 mHa)")
tag = PASS if floor_mHa < 0.01 else (WARN if floor_mHa < 1.0 else FAIL)
print(f"  DexcG floor vs CASCI  : {floor_mHa:.6f} mHa{tag}")

# ── FACT 6: k-UpCCGSD parameter count ───────────────────────────────────────
print("\n── FACT 6: k-UpCCGSD parameter count ──────────────────────────────────")
qc_k = ansatze["k-UpCCGSD"]
print(f"  k-UpCCGSD num_parameters = {qc_k.num_parameters}  (spec: 9){PASS if qc_k.num_parameters==9 else FAIL}")
qc_uccsd = ansatze["UCCSD"]
print(f"  UCCSD     num_parameters = {qc_uccsd.num_parameters}  (spec: 3){PASS if qc_uccsd.num_parameters==3 else FAIL}")

# ── FACT 7: seed not used in random init ─────────────────────────────────────
print("\n── FACT 7: 'random' init range ────────────────────────────────────────")
pts = [get_initial_point("random", 5, seed=s) for s in [0, 1, 2, 3, 42]]
all_in_01 = all(np.all((p >= 0) & (p <= 1.0)) for p in pts)
print(f"  random init in [0,1]  : {all_in_01}  (spec/README claims U(0,2pi) — BUG confirmed if True){WARN if all_in_01 else PASS}")
print(f"  Sample seed=42: {pts[-1]}")

# ── FACT 8: README mismatch flags ────────────────────────────────────────────
print("\n── FACT 8: README / code mismatches (manual check) ────────────────────")
print(f"  PCU2 params: {n_pcu2}  (README likely says 16 → {'BUG' if n_pcu2 != 16 else 'OK'})")
print(f"  k-UpCCGSD params: {qc_k.num_parameters}  (README likely says > 9)")

# ── FACT 9: hardware.py fabricated fallbacks ─────────────────────────────────
print("\n── FACT 9: hardware.py fabricated fallback values ─────────────────────")
hw_src = Path(__file__).parent.parent / "src" / "hardware.py"
hw_text = hw_src.read_text(encoding="utf-8")
has_fab1 = "0.00178" in hw_text
has_fab2 = "0.00185" in hw_text
has_fab3 = "0.00045" in hw_text
print(f"  'exact+0.00178' in hardware.py : {has_fab1}  {WARN if has_fab1 else PASS}")
print(f"  'exact+0.00185' in hardware.py : {has_fab2}  {WARN if has_fab2 else PASS}")
print(f"  'exact+0.00045' in hardware.py : {has_fab3}  {WARN if has_fab3 else PASS}")

print()
print(SEP)
print("  BASELINE VERIFICATION COMPLETE")
print(SEP)

