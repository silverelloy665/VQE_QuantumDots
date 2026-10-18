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

from qiskit.circuit.library import XXPlusYYGate

def build_particle_conserving_u2(
    num_spatial_orbitals: int = 2,
    num_particles: tuple[int, int] = (1, 1),
    reps: int = 2,
    qubit_mapper: JordanWignerMapper | None = None
) -> QuantumCircuit:
    """
    Constructs the genuinely Particle- and Sz-conserving U2 ansatz (ParticleConservingU2):
    - Hartree-Fock reference state (|0101> for 2e, 2o)
    - For each layer (reps):
        - RZ rotation on every qubit (preserves particle number on every individual qubit)
        - Spin-aware Givens rotations (XX+YY) on orbital pairs within the alpha and beta sectors:
            - alpha pair (0, 1) preserves N_alpha = 1
            - beta pair (2, 3) preserves N_beta = 1
        - Double-excitation block between alpha and beta sectors, preserving both N=2 and Sz=0
          (enables variational access to the ground state beyond product-state correlation)
    For 2e/2o with reps=2:
        4 RZ + 2 Givens (alpha, beta) + 1 Double excitation = 7 parameters per rep.
        Total parameters = 2 * 7 = 14 parameters.
    100% of state probability amplitude remains strictly inside the (N_alpha=1, N_beta=1) sector.
    """
    mapper = qubit_mapper or JordanWignerMapper()
    num_qubits = num_spatial_orbitals * 2
    
    # 4 RZ + 2 Givens (alpha 0-1, beta 2-3) + 1 Double excitation = 7 params per layer
    params_per_rep = num_qubits + 2 + 1
    total_params = reps * params_per_rep
    
    params = ParameterVector("θ_PCU2", total_params)
    qc = QuantumCircuit(num_qubits, name="ParticleConservingU2")
    
    # Hartree-Fock initial state
    hf_state = HartreeFock(num_spatial_orbitals, num_particles, mapper)
    qc.compose(hf_state, inplace=True)
    
    dexc_template = UCC(
        num_spatial_orbitals=num_spatial_orbitals,
        num_particles=num_particles,
        excitations="d",
        qubit_mapper=mapper,
        initial_state=None
    )
    
    param_idx = 0
    for r in range(reps):
        # 1. RZ on every qubit (preserves local particle number)
        for q in range(num_qubits):
            qc.rz(params[param_idx], q)
            param_idx += 1
            
        # 2. Spin-aware Givens rotations: alpha pair (0, 1) and beta pair (2, 3)
        qc.append(XXPlusYYGate(params[param_idx]), [0, 1])
        param_idx += 1
        qc.append(XXPlusYYGate(params[param_idx]), [2, 3])
        param_idx += 1
        
        # 3. Double excitation block coupling alpha and beta pairs
        d_block = dexc_template.assign_parameters([params[param_idx]])
        qc.compose(d_block, inplace=True)
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
