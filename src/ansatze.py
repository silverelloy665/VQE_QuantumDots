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

import qiskit
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

def generate_upccgsd_excitations(num_spatial_orbitals: int, num_particles: tuple[int, int]):
    """
    Generates paired doubles + generalized singles excitations for UpCCGSD.
    Generic in (num_spatial_orbitals, num_particles).
    
    Generates:
      1. Generalized singles (all p < q for alpha and beta spins)
      2. Paired doubles (p_alpha p_beta -> q_alpha q_beta for all spatial pairs p < q)
    """
    n_so = num_spatial_orbitals
    excitations = []
    # 1. Generalized singles (all pairs p < q for alpha and beta)
    for p in range(n_so):
        for q in range(p + 1, n_so):
            excitations.append(((p,), (q,)))                  # alpha spin
            excitations.append(((p + n_so,), (q + n_so,)))    # beta spin
    # 2. Paired doubles (p_alpha p_beta -> q_alpha q_beta for all p < q)
    for p in range(n_so):
        for q in range(p + 1, n_so):
            excitations.append(((p, p + n_so), (q, q + n_so)))
    return excitations

def get_ansatz_dict(
    num_spatial_orbitals: int = 2,
    num_particles: tuple[int, int] = (1, 1),
    k_reps: int = 3,
    pcu2_reps: int = 2,
    include_sa: bool = True
) -> dict[str, QuantumCircuit]:
    """
    Returns the dictionary of benchmark ansätze:
    - DexcG: UCC with excitations='d' (1 parameter for 2e/2o)
    - PCU2: ParticleConservingU2 with genuine Givens rotations and double excitation layer (14 params for 2e/2o)
    - UCCSD: Standard UCC singles and doubles (3 parameters for 2e/2o)
    - UCCSD-SA: Spin-adapted UCCSD with tied single excitations theta_alpha = theta_beta (2 parameters for 2e/2o)
    - k-UpCCGSD: Generic UpCCGSD with paired doubles + generalized singles, reps=k (9 parameters for 2e/2o, k=3).
      Note: For (2e, 2o), k-UpCCGSD is strictly redundant with UCCSD as generalized singles/doubles coincide with UCCSD.
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
    
    # Generic UpCCGSD with paired doubles + generalized singles
    kupccgsd = UCC(
        num_spatial_orbitals=num_spatial_orbitals,
        num_particles=num_particles,
        excitations=generate_upccgsd_excitations,
        qubit_mapper=mapper,
        initial_state=hf_state,
        reps=k_reps
    )
    kupccgsd.name = "k-UpCCGSD"
    
    # Attach redundancy metadata for 2e,2o space
    is_2e2o = (num_spatial_orbitals == 2 and num_particles == (1, 1))
    setattr(kupccgsd, "redundant_with_uccsd", is_2e2o)
    
    ansatze = {
        "DexcG": dexcg,
        "PCU2": pcu2,
        "UCCSD": uccsd,
    }
    
    if include_sa and is_2e2o:
        # Spin-adapted UCCSD (theta_alpha = theta_beta tied) -> 2 parameters
        theta_sa = ParameterVector("theta_SA", 2)
        uccsd_sa = uccsd.assign_parameters({
            uccsd.parameters[0]: theta_sa[0],
            uccsd.parameters[1]: theta_sa[0],
            uccsd.parameters[2]: theta_sa[1]
        })
        uccsd_sa.name = "UCCSD-SA"
        ansatze["UCCSD-SA"] = uccsd_sa
        
    ansatze["k-UpCCGSD"] = kupccgsd
    return ansatze

def evaluate_parity_mapper_reduction(basis: str = "6-31g(d,p)") -> dict:
    """
    Evaluates ParityMapper with 2-qubit reduction for the (2e, 2o) active space.
    Compares spectrum with CASCI exact ground state and counts resulting CNOT/qubit metrics.
    """
    import json
    from qiskit_nature.second_q.mappers import ParityMapper
    from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
    from src.molecule import DATA_DIR
    
    clean_basis = "631gdp" if "6-31g" in basis.lower() else "sto3g"
    cache_path = DATA_DIR / f"bn_dot_{clean_basis}.json"
    with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    h1 = np.array(data["h1_active"])
    h2 = np.array(data["h2_active"])
    h2_phys = np.einsum("pqrs->prqs", h2)
    ee = ElectronicEnergy.from_raw_integrals(h1, h2_phys)
    ferm_op = ee.second_q_op()
    
    parity_mapper = ParityMapper(num_particles=(1, 1))
    qubit_op_parity = parity_mapper.map(ferm_op)
    
    evals, _ = np.linalg.eigh(qubit_op_parity.to_matrix())
    e_core = float(data.get("e_core", data.get("inactive_energy")))
    ground_parity = float(evals[0]) + e_core
    casci_exact = float(data["casci_energy"])
    
    spectrum_matches = np.isclose(ground_parity, casci_exact, atol=1e-6)
    
    hf_parity = HartreeFock(2, (1, 1), parity_mapper)
    dexcg_p = UCC(2, (1, 1), "d", parity_mapper, initial_state=hf_parity).decompose().decompose()
    uccsd_p = UCC(2, (1, 1), "sd", parity_mapper, initial_state=hf_parity).decompose().decompose()
    
    cnot_dexcg = dict(dexcg_p.count_ops()).get("cx", 0)
    cnot_uccsd = dict(uccsd_p.count_ops()).get("cx", 0)
    
    return {
        "parity_num_qubits": qubit_op_parity.num_qubits,
        "parity_pauli_terms": len(qubit_op_parity),
        "parity_ground_energy": ground_parity,
        "casci_exact_energy": casci_exact,
        "spectrum_matches": spectrum_matches,
        "difference_Ha": abs(ground_parity - casci_exact),
        "dexcg_parity_cnot": cnot_dexcg,
        "dexcg_parity_depth": dexcg_p.depth(),
        "uccsd_parity_cnot": cnot_uccsd,
        "uccsd_parity_depth": uccsd_p.depth(),
        "status": "SPECTRUM VERIFIED MATCH" if spectrum_matches else "SPECTRUM MISMATCH (NOT ADOPTED)"
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
        from src.config import get_target_backend
        backend = get_target_backend()
        
    records = []
    seeds = [0, 1, 2, 42, 100]
    
    for name, qc in ansatze_dict.items():
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
            
        # Transpilation analysis across multiple transpiler seeds to minimize 2Q gate count
        best_tqc = None
        best_2q = float("inf")
        best_seed = seeds[0]
        
        for s in seeds:
            try:
                pm = generate_preset_pass_manager(backend=backend, optimization_level=3, seed_transpiler=s)
                tqc = pm.run(decomposed)
                ops_s = tqc.count_ops()
                n2q_s = sum(count for op, count in ops_s.items() if op in ["cx", "ecr", "cz"])
                if n2q_s < best_2q:
                    best_2q = n2q_s
                    best_seed = s
                    best_tqc = tqc
            except Exception:
                continue
                
        if best_tqc is None:
            pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
            best_tqc = pm.run(decomposed)
            ops_s = best_tqc.count_ops()
            best_2q = sum(count for op, count in ops_s.items() if op in ["cx", "ecr", "cz"])
            best_seed = 0
            
        t_ops = best_tqc.count_ops()
        
        records.append({
            "Ansatz": name,
            "Parameters": qc.num_parameters,
            "Raw Depth": decomposed.depth(),
            "Raw 1-Qubit Gates": num_1q_gates_raw,
            "Raw 2-Qubit Gates": num_2q_gates_raw,
            "Transpiled Depth": best_tqc.depth(),
            "Transpiled 2-Qubit Gates": best_2q,
            "Best_Transpiled_2Q_Gates": best_2q,
            "Transpiler_Seed": best_seed,
            "Target_Backend": getattr(backend, "name", "unknown"),
            "Qiskit_Version": qiskit.__version__,
            "Transpiled Total Gates": sum(t_ops.values())
        })
        
    df = pd.DataFrame(records)
    return df

if __name__ == "__main__":
    df = analyze_and_render_circuits()
    print(df.to_string(index=False))
