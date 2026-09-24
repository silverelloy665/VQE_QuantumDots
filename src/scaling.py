"""
Qubit and Parameter Scaling Analysis for B8N8H10.
Computes analytical scaling of qubits, UCCSD singles, and UCCSD doubles
for full-molecule and active-space configurations under Jordan-Wigner mapping.
"""
import math
import pandas as pd


def compute_uccsd_counts(n_electrons: int, n_spatial_orbitals: int) -> dict:
    """
    Analytically computes Jordan-Wigner qubits, singles, doubles, and total UCCSD
    parameter counts for closed-shell singlet (spin-conserving alpha/beta excitations).

    Formulas:
        o = n_electrons // 2 (occupied spatial orbitals)
        v = n_spatial_orbitals - o (virtual spatial orbitals)
        qubits = 2 * n_spatial_orbitals
        singles = 2 * o * v (alpha->alpha, beta->beta)
        doubles = 2 * C(o, 2) * C(v, 2) + (o * v)^2
                (alpha,alpha->alpha,alpha; beta,beta->beta,beta; alpha,beta->alpha,beta)
    """
    o = n_electrons // 2
    v = n_spatial_orbitals - o
    qubits = 2 * n_spatial_orbitals

    if o < 1 or v < 1:
        return {
            "occupied": o,
            "virtual": v,
            "qubits": qubits,
            "singles": 0,
            "doubles": 0,
            "total_uccsd": 0
        }

    singles = 2 * o * v
    doubles_same_spin = 2 * math.comb(o, 2) * math.comb(v, 2)
    doubles_mixed_spin = (o * v) ** 2
    doubles = doubles_same_spin + doubles_mixed_spin
    total = singles + doubles

    return {
        "occupied": o,
        "virtual": v,
        "qubits": qubits,
        "singles": singles,
        "doubles": doubles,
        "total_uccsd": total
    }


def get_qubit_and_parameter_scaling_table() -> pd.DataFrame:
    """
    Returns the analytical scaling comparison table across:
    - Full Molecule STO-3G (90 spatial AOs, 106 electrons)
    - Full Molecule 6-31G(d,p) (274 spherical spatial AOs, 106 electrons)
    - Active Space (2e, 2o) (4 qubits)
    - Active Space (4e, 4o) (8 qubits)
    - Active Space (6e, 6o) (12 qubits)
    """
    systems = [
        ("Full Molecule (STO-3G)", 106, 90, "Full System"),
        ("Full Molecule (6-31G(d,p))", 106, 274, "Full System"),
        ("Active Space (2e, 2o)", 2, 2, "Active Space (Benchmark)"),
        ("Active Space (4e, 4o)", 4, 4, "Active Space (Extended)"),
        ("Active Space (6e, 6o)", 6, 6, "Active Space (Extended)"),
    ]

    records = []
    for label, n_e, n_o, category in systems:
        counts = compute_uccsd_counts(n_e, n_o)
        records.append({
            "Configuration": label,
            "Category": category,
            "Electrons": n_e,
            "Spatial Orbitals": n_o,
            "JW Qubits": counts["qubits"],
            "UCCSD Singles": counts["singles"],
            "UCCSD Doubles": counts["doubles"],
            "Total UCCSD Params": counts["total_uccsd"],
            "Feasible on NISQ/Simulator": "Yes" if counts["qubits"] <= 12 else "No (Qubit Bottleneck)"
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    df = get_qubit_and_parameter_scaling_table()
    print("==========================================================================================")
    print("ANALYTICAL QUBIT AND UCCSD PARAMETER SCALING FOR B8N8H10")
    print("==========================================================================================")
    print(df.to_string(index=False))

