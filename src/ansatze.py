"""
Ansatz definitions, circuit gallery generator, and transpilation analysis for BN Quantum Dot.
"""
import warnings
from pathlib import Path
from collections import OrderedDict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse import SparseEfficiencyWarning

from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit_nature.second_q.circuit.library import HartreeFock, UCC
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.providers.fake_provider import GenericBackendV2

warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def build_particle_conserving_u2(
    num_spatial_orbitals: int = 2,
    num_particles: tuple[int, int] = (1, 1),
    reps: int = 2,
    qubit_mapper: JordanWignerMapper | None = None
) -> QuantumCircuit:
    """
    Constructs the hand-built ParticleConservingU2 (PCU2) ansatz:
    - Hartree-Fock initial state
    - Hartree-Fock initial state (|0101>)
    - For each layer (reps):
        - RZ rotation on every qubit
        - Even-pair entanglers on (0,1), (2,3), ... : CNOT - CRX - CNOT
        - Odd-pair entanglers on (1,2), (3,4), ...  : CNOT - CRX - CNOT
        - RZ rotation on every qubit (4 params)
        - Even-pair entanglers on (0,1), (2,3): CNOT - CRX - CNOT (2 params)
        - Odd-pair entanglers on (1,2): CNOT - CRX - CNOT (1 param)
    Total parameters per rep = 4 + 2 + 1 = 7. For reps=2, total parameters = 14.
    For 2e/2o with reps=2, total parameters = 2 * (4 + 2 + 1) = 14.
    """
    mapper = qubit_mapper or JordanWignerMapper()
    num_qubits = num_spatial_orbitals * 2
    
    # 1 RZ per qubit + 1 CRX per even pair + 1 CRX per odd pair
    n_even_pairs = len(range(0, num_qubits - 1, 2))
    n_odd_pairs = len(range(1, num_qubits - 1, 2))
    params_per_rep = num_qubits + n_even_pairs + n_odd_pairs
    total_params = reps * params_per_rep
    
    params = ParameterVector("θ_PCU2", total_params)
    qc = QuantumCircuit(num_qubits, name="PCU2")
    
    # Hartree-Fock initial state
    hf_state = HartreeFock(num_spatial_orbitals, num_particles, mapper)
    qc.compose(hf_state, inplace=True)
    
    param_idx = 0
    for r in range(reps):
        # 1. RZ on every qubit
        for q in range(num_qubits):
            qc.rz(params[param_idx], q)
            param_idx += 1
            
        # 2. Even pair entanglers: CNOT - CRX - CNOT
        for q in range(0, num_qubits - 1, 2):
            qc.cx(q, q + 1)
            qc.crx(params[param_idx], q, q + 1)
            qc.cx(q, q + 1)
            param_idx += 1
            
        # 3. Odd pair entanglers: CNOT - CRX - CNOT
        for q in range(1, num_qubits - 1, 2):
            qc.cx(q, q + 1)
            qc.crx(params[param_idx], q, q + 1)
            qc.cx(q, q + 1)
            param_idx += 1
            
    return qc

def get_ansatz_dict(
    num_spatial_orbitals: int = 2,
    num_particles: tuple[int, int] = (1, 1),
    k_reps: int = 3,
    pcu2_reps: int = 2
) -> dict[str, QuantumCircuit]:
    """
    Returns the dictionary of the 4 benchmark ansätze:
    - DexcG: UCC with excitations='d'
    - PCU2: ParticleConservingU2 (2 reps)
    - PCU2: ParticleConservingU2 (reps=pcu2_reps)
    - UCCSD: UCC with excitations='sd'
    - k-UpCCGSD: UCC with excitations='sd', generalized=True, reps=k (k=3)
    - DexcG: UCC with excitations='d' (1 parameter for 2e/2o)
    - PCU2: ParticleConservingU2 (reps=2, 14 parameters for 2e/2o)
    - UCCSD: UCC with excitations='sd' (3 parameters for 2e/2o)
    - k-UpCCGSD: Generalized UCC with excitations='sd', generalized=True, reps=k (k=3, 9 parameters for 2e/2o).
    - k-UpCCGSD: Generalized UCC with excitations='sd', generalized=True, reps=k_reps.
      Note: This is a generalized unitary coupled cluster ansatz with k repetitions and is not pair-restricted.
    """
    mapper = JordanWignerMapper()
    hf_state = HartreeFock(num_spatial_orbitals, num_particles, mapper)
    
    dexcg = UCC(
        num_spatial_orbitals=num_spatial_orbitals,
        num_particles=num_particles,
        excitations="d",
        qubit_mapper=mapper,
        initial_state=hf_state
    )
    dexcg.name = "DexcG"
    
    pcu2 = build_particle_conserving_u2(
        num_spatial_orbitals=num_spatial_orbitals,
        num_particles=num_particles,
        reps=pcu2_reps,
        qubit_mapper=mapper
    )
    
    uccsd = UCC(
        num_spatial_orbitals=num_spatial_orbitals,
        num_particles=num_particles,
        excitations="sd",
        qubit_mapper=mapper,
        initial_state=hf_state
    )
    uccsd.name = "UCCSD"
    
    kupccgsd = UCC(
        num_spatial_orbitals=num_spatial_orbitals,
        num_particles=num_particles,
        excitations="sd",
        generalized=True,
        reps=k_reps,
        qubit_mapper=mapper,
        initial_state=hf_state
    )
    kupccgsd.name = "k-UpCCGSD"
    
    return {
        "DexcG": dexcg,
        "PCU2": pcu2,
        "UCCSD": uccsd,
        "k-UpCCGSD": kupccgsd
    }

def analyze_and_render_circuits(
    ansatze_dict: dict[str, QuantumCircuit] | None = None,
    backend=None,
    save_pngs: bool = True
) -> pd.DataFrame:
    """
    Renders decomposed circuits with matplotlib, saves PNGs to figures/,
    and returns a summary DataFrame of parameters, depth, raw gates,
    and transpiled metrics.
    """
    if ansatze_dict is None:
        ansatze_dict = get_ansatz_dict()
        
    if backend is None:
        backend = GenericBackendV2(num_qubits=5)
        
    pm = generate_preset_pass_manager(backend=backend, optimization_level=2)
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    records = []
    
    for name, qc in ansatze_dict.items():
        # Decompose 2-3 levels so elementary gates (CNOT, RZ, RX, etc.) are visible
        # Decompose 2 levels so elementary gates (CNOT, RZ, RX, etc.) are visible
        decomposed = qc.decompose()
        if any(op.name in ["PauliEvolution", "HartreeFock"] for op in decomposed.data):
            decomposed = decomposed.decompose()
            
        raw_ops = decomposed.count_ops()
        num_2q_gates_raw = sum(count for op, count in raw_ops.items() if op in ["cx", "crx", "cz", "ecr"])
        num_1q_gates_raw = sum(count for op, count in raw_ops.items() if op not in ["cx", "crx", "cz", "ecr"])
        
        # Save circuit diagram to figures/
        if save_pngs:
            fig_path = FIGURES_DIR / f"circuit_{name}.png"
            fig = decomposed.draw(output="mpl", style="iqp", fold=20)
            fig.savefig(fig_path, bbox_inches="tight", dpi=300)
            plt.close(fig)
            
        # Transpilation analysis
        # Transpilation analysis (unbound symbolic parameters)
        transpiled = pm.run(decomposed)
        t_ops = transpiled.count_ops()
        t_2q_gates = sum(count for op, count in t_ops.items() if op in ["cx", "ecr", "cz"])
        
        records.append({
            "Ansatz": name,
            "Parameters": qc.num_parameters,
            "Raw Depth": decomposed.depth(),
            "Raw 1-Qubit Gates": num_1q_gates_raw,
            "Raw 2-Qubit Gates": num_2q_gates_raw,
            "Transpiled Depth": transpiled.depth(),
            "Transpiled 2-Qubit Gates": t_2q_gates,
            "Transpiled Total Gates": sum(t_ops.values())
        })
        
    df = pd.DataFrame(records)
    return df

if __name__ == "__main__":
    df = analyze_and_render_circuits()
    print(df.to_string(index=False))
