import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.ansatze import get_ansatz_dict

FIGURES_DIR = REPO_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print("Rendering hard-problem (8 qubits, 4e4o) circuit diagrams...")

hard_ansatze = get_ansatz_dict(
    num_spatial_orbitals=4,
    num_particles=(2, 2),
    pcu2_reps=1,
    include_k_variants=True,
    include_sa=True,
    include_hf=True
)

for name, qc in hard_ansatze.items():
    if qc.num_parameters == 0:
        continue
    decomposed = qc.decompose()
    if any(op.name in ["PauliEvolution", "HartreeFock"] for op in decomposed.data):
        decomposed = decomposed.decompose()

    fig_path = FIGURES_DIR / f"circuit_{name}.png"
    fig_path_8q = FIGURES_DIR / f"circuit_{name}_4e4o_8q.png"

    # Draw circuit diagram
    fig = decomposed.draw(output="mpl", style="iqp", fold=25)
    fig.savefig(fig_path, bbox_inches="tight", dpi=200)
    fig.savefig(fig_path_8q, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Rendered {name}: saved to {fig_path.name} and {fig_path_8q.name} (qubits: {decomposed.num_qubits}, depth: {decomposed.depth()})")

print("All hard-problem circuit diagrams successfully generated.")

