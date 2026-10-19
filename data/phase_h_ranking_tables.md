# Round 3 VQE Benchmark Ranking Tables (Hard Problem: B8N8H10 8-Qubit 4e4o)
> **Benchmark Context**: $E_{\text{HF}} = -637.573196\text{ Ha}$ (Reference determinant), $E_{\text{CASCI}} = -637.582468\text{ Ha}$, $\Delta E_{\text{corr}} = 8.1392\text{ mHa}$.

## 1. Zero Initialization (Deterministic: $\theta = 0$)
*Evaluates ansatz expressiveness and local gradient landscape when starting from the Hartree-Fock chemical reference state.*

| Ansatz | Optimizer | Error (mHa) | % Corr Recovered | Evaluations | Converged | Wall Time (s) |
| --- | --- | --- | --- | --- | --- | --- |
| 3-UpCCGSD | L-BFGS-B | 0.0010 | 100.0% | 35643 | No | 305.7 |
| 2-UpCCGSD | L-BFGS-B | 0.0064 | 99.9% | 22849 | Yes | 95.9 |
| UCCSD-SA | L-BFGS-B | 0.0078 | 99.9% | 405 | Yes | 4.2 |
| UCCSD | L-BFGS-B | 0.0078 | 99.9% | 848 | Yes | 7.5 |
| PCU2 | ADAM | 0.0148 | 99.8% | 11181 | Yes | 513.8 |
| DexcG | L-BFGS-B | 0.0149 | 99.8% | 407 | Yes | 19.4 |
| DexcG | ADAM | 0.0149 | 99.8% | 11101 | Yes | 476.9 |
| PCU2 | L-BFGS-B | 0.0149 | 99.8% | 650 | Yes | 32.1 |
| DexcG | GD | 0.0149 | 99.8% | 630 | Yes | 25.3 |
| DexcG | COBYLA | 0.0149 | 99.8% | 263 | Yes | 20.4 |
| PCU2 | COBYLA | 0.0149 | 99.8% | 734 | Yes | 56.1 |
| 1-UpCCGSD | L-BFGS-B | 1.4836 | 81.8% | 1443 | Yes | 5.5 |
| DexcG | QNSPSA | 3.1610 | 61.2% | 929 | No | 434.0 |
| DexcG | ROTOSOLVE | 4.1879 | 48.5% | 216 | Yes | 25.8 |
| HF_baseline | none | 8.1392 | -0.0% | 1 | Yes | 0.0 |
| DexcG | SPSA | 55.1673 | -577.8% | 903 | No | 55.0 |

## 2. Perturbed Initializations (Combined $\sigma \in \{0.01, 0.05, 0.1, 0.3\}$, 40 runs per config)
*Evaluates optimizer basin stability when starting from chemically informed parameters with realistic perturbation noise.*

| Ansatz | Optimizer | N | Median Error (mHa) | IQR (mHa) | 95% CI (mHa) | Median % Corr | Success Rate | Median Evals |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3-UpCCGSD | L-BFGS-B | 5 | 0.0044 | 0.0005 | [0.0043, 0.0069] | 99.9% | 100.0% | 35207 |
| 2-UpCCGSD | L-BFGS-B | 40 | 0.0059 | 0.0035 | [0.0050, 0.0079] | 99.9% | 100.0% | 23360 |
| UCCSD-SA | L-BFGS-B | 40 | 0.0078 | 0.0000 | [0.0078, 0.0078] | 99.9% | 97.5% | 495 |
| UCCSD | L-BFGS-B | 40 | 0.0078 | 0.0000 | [0.0078, 0.0078] | 99.9% | 97.5% | 742 |
| PCU2 | ADAM | 29 | 0.0142 | 0.0023 | [0.0131, 0.0146] | 99.8% | 100.0% | 10921 |
| PCU2 | L-BFGS-B | 40 | 0.0149 | 0.0000 | [0.0149, 0.0149] | 99.8% | 100.0% | 1235 |
| DexcG | L-BFGS-B | 40 | 0.0149 | 0.0000 | [0.0149, 0.0149] | 99.8% | 100.0% | 370 |
| DexcG | ADAM | 40 | 0.0149 | 0.0000 | [0.0149, 0.0149] | 99.8% | 100.0% | 11101 |
| DexcG | GD | 40 | 0.0149 | 0.0000 | [0.0149, 0.0149] | 99.8% | 100.0% | 815 |
| PCU2 | COBYLA | 40 | 0.0149 | 0.0000 | [0.0149, 0.0149] | 99.8% | 100.0% | 860 |
| DexcG | COBYLA | 40 | 0.0149 | 0.0000 | [0.0149, 0.0149] | 99.8% | 100.0% | 283 |
| 1-UpCCGSD | L-BFGS-B | 40 | 1.4836 | 0.0000 | [1.4836, 1.4836] | 81.8% | 100.0% | 1554 |
| DexcG | ROTOSOLVE | 40 | 7.4420 | 6.9794 | [6.3640, 9.9235] | 8.6% | 0.0% | 216 |
| DexcG | QNSPSA | 40 | 39.3155 | 126.3235 | [18.3861, 68.7953] | -383.0% | 0.0% | 929 |
| DexcG | SPSA | 40 | 49.1389 | 6.4113 | [46.5982, 50.1371] | -503.7% | 0.0% | 903 |

## 3. Random Initialization ($U(0, 2\pi)^N$, 10 seeds)
*Evaluates global optimization ability and resistance to barren plateaus / local minima in the absence of chemical insight.*

| Ansatz | Optimizer | N | Median Error (mHa) | IQR (mHa) | 95% CI (mHa) | Median % Corr | Success Rate | Median Evals |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DexcG | L-BFGS-B | 10 | 0.0149 | 0.0000 | [0.0149, 0.0149] | 99.8% | 100.0% | 3589 |
| 2-UpCCGSD | L-BFGS-B | 10 | 0.0161 | 0.0077 | [0.0101, 0.0202] | 99.8% | 100.0% | 23433 |
| DexcG | ADAM | 10 | 0.0406 | 0.1655 | [0.0294, 0.2455] | 99.5% | 100.0% | 11101 |
| 1-UpCCGSD | L-BFGS-B | 10 | 1.4837 | 1.3982 | [0.0835, 1.4850] | 81.8% | 100.0% | 7548 |
| UCCSD-SA | L-BFGS-B | 10 | 2.4627 | 0.5956 | [1.1215, 2.8051] | 69.7% | 20.0% | 14805 |
| UCCSD | L-BFGS-B | 10 | 2.5568 | 2.8451 | [0.0106, 3.0214] | 68.6% | 40.0% | 17410 |
| PCU2 | L-BFGS-B | 10 | 3.0203 | 0.7054 | [2.3462, 135.4695] | 62.9% | 0.0% | 21027 |
| PCU2 | COBYLA | 10 | 5.0071 | 266.7119 | [4.6279, 272.4796] | 38.5% | 0.0% | 1001 |
| DexcG | COBYLA | 10 | 532.3723 | 20.9460 | [265.5447, 555.0935] | -6440.9% | 20.0% | 1001 |
| DexcG | GD | 10 | 594.0371 | 28.6382 | [573.4921, 612.9819] | -7198.5% | 0.0% | 11101 |
| DexcG | QNSPSA | 10 | 600.2829 | 21.2235 | [588.1618, 619.7126] | -7275.2% | 0.0% | 929 |
| DexcG | ROTOSOLVE | 10 | 607.2438 | 27.5817 | [596.5987, 628.3674] | -7360.8% | 0.0% | 216 |
| DexcG | SPSA | 10 | 613.1759 | 39.6859 | [603.0025, 645.6884] | -7433.6% | 0.0% | 903 |
