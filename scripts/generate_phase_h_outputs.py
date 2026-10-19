import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.molecule import load_hard_hamiltonian, DATA_DIR

CACHE_FILE = DATA_DIR / "benchmark_results_hard.json"
SUMMARY_CSV = DATA_DIR / "hard_benchmark_summary.csv"
TABLES_MD = DATA_DIR / "phase_h_ranking_tables.md"
FIGURES_DIR = REPO_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def compute_bootstrap_median_ci(data: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, seed: int = 42) -> tuple[float, float]:
    """Computes non-parametric bootstrap confidence interval for the median."""
    n = len(data)
    if n <= 1:
        val = float(data[0]) if n == 1 else 0.0
        return val, val
    rng = np.random.RandomState(seed)
    boot_medians = [
        float(np.median(rng.choice(data, size=n, replace=True)))
        for _ in range(n_boot)
    ]
    low = float(np.percentile(boot_medians, 100.0 * (alpha / 2.0)))
    high = float(np.percentile(boot_medians, 100.0 * (1.0 - alpha / 2.0)))
    return low, high


def to_markdown_table(df: pd.DataFrame) -> str:
    """Formats a DataFrame as a GitHub-flavored Markdown table without tabulate."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |"]
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def generate_phase_h_analysis():
    print("=" * 80)
    print("PHASE H: GENERATING HONEST RANKING TABLES AND FIGURES")
    print("=" * 80)

    if not CACHE_FILE.exists():
        raise FileNotFoundError(f"Cache file {CACHE_FILE} not found.")

    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cdata = json.load(f)
    records = cdata.get("records", [])
    print(f"Loaded {len(records)} benchmark records from Schema v3.0 cache.")

    _, exact_energy, meta = load_hard_hamiltonian()
    ref_det_energy = float(meta["reference_determinant_energy"])
    delta_corr = abs(ref_det_energy - exact_energy)

    # Process all records into structured DataFrame
    processed = []
    for r in records:
        f_e = float(r["final_energy"])
        err_mha = abs(f_e - exact_energy) * 1000.0
        pct_corr = ((ref_det_energy - f_e) / delta_corr) * 100.0 if delta_corr > 1e-12 else 0.0
        better_than_hf = (f_e < ref_det_energy - 1e-9)
        succ = (err_mha < 1.6 and better_than_hf)

        init_raw = r["init"]
        if init_raw.startswith("perturbed"):
            init_group = "perturbed"
            sigma = float(init_raw.split("_")[1]) if "_" in init_raw else 0.05
        else:
            init_group = init_raw
            sigma = 0.0

        processed.append({
            "ansatz": r["ansatz"],
            "optimizer": r["optimizer"],
            "init": init_raw,
            "init_group": init_group,
            "sigma": sigma,
            "seed": r.get("seed", 0),
            "final_energy": f_e,
            "error_mHa": err_mha,
            "pct_corr_recovered": pct_corr,
            "better_than_HF": better_than_hf,
            "success": succ,
            "converged": bool(r.get("converged", False)),
            "hit_maxiter": bool(r.get("hit_maxiter", False)),
            "total_evaluations": int(r.get("total_evaluations", 0)),
            "iterations_used": int(r.get("iterations_used", 0)),
            "wall_time": float(r.get("wall_time", 0.0))
        })

    df = pd.DataFrame(processed)

    # Generate hard_benchmark_summary.csv
    summary_rows = []
    groups = df.groupby(["init", "ansatz", "optimizer"])
    for (init_name, a_name, opt_name), group in groups:
        errors = group["error_mHa"].values
        pcts = group["pct_corr_recovered"].values
        evals = group["total_evaluations"].values
        convs = group["converged"].values
        succs = group["success"].values
        betters = group["better_than_HF"].values
        times = group["wall_time"].values

        n_runs = len(errors)
        q25, med, q75 = np.percentile(errors, [25, 50, 75])
        iqr = q75 - q25
        min_err = np.min(errors)
        max_err = np.max(errors)
        med_pct = np.median(pcts)
        med_evals = np.median(evals)
        frac_conv = np.mean(convs) * 100.0
        succ_rate = np.mean(succs) * 100.0
        better_rate = np.mean(betters) * 100.0
        mean_time = np.mean(times)

        ci_low, ci_high = compute_bootstrap_median_ci(errors)

        summary_rows.append({
            "Init_Type": init_name,
            "Ansatz": a_name,
            "Optimizer": opt_name,
            "N_Runs": n_runs,
            "Median_Error_mHa": float(med),
            "IQR_Error_mHa": float(iqr),
            "CI_95_Lower_mHa": ci_low,
            "CI_95_Upper_mHa": ci_high,
            "Min_Error_mHa": float(min_err),
            "Max_Error_mHa": float(max_err),
            "Median_Pct_Corr": float(med_pct),
            "Success_Rate_pct": float(succ_rate),
            "Better_than_HF_pct": float(better_rate),
            "Fraction_Converged_pct": float(frac_conv),
            "Median_Evaluations": float(med_evals),
            "Mean_Wall_Time_s": float(mean_time)
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.sort_values(by=["Init_Type", "Median_Error_mHa"], inplace=True)
    df_summary.to_csv(SUMMARY_CSV, index=False)
    print(f"Generated {SUMMARY_CSV} with {len(df_summary)} aggregated rows.")

    # Generate Markdown Tables per initialization regime
    md_output = []
    md_output.append("# Round 3 VQE Benchmark Ranking Tables (Hard Problem: B8N8H10 8-Qubit 4e4o)\n")
    md_output.append("> **Benchmark Context**: $E_{\\text{HF}} = -637.573196\\text{ Ha}$ (Reference determinant), $E_{\\text{CASCI}} = -637.582468\\text{ Ha}$, $\\Delta E_{\\text{corr}} = 8.1392\\text{ mHa}$.\n")

    # 1. Zero Initialization (Deterministic, Ansatz Expressiveness from Chemical Reference)
    md_output.append("\n## 1. Zero Initialization (Deterministic: $\\theta = 0$)\n")
    md_output.append("*Evaluates ansatz expressiveness and local gradient landscape when starting from the Hartree-Fock chemical reference state.*\n\n")
    sub_zero = df[df["init"] == "zero"].sort_values(by="error_mHa")
    zero_table = []
    for _, r in sub_zero.iterrows():
        zero_table.append({
            "Ansatz": r["ansatz"],
            "Optimizer": r["optimizer"],
            "Error (mHa)": f"{r['error_mHa']:.4f}",
            "% Corr Recovered": f"{r['pct_corr_recovered']:.1f}%",
            "Evaluations": r["total_evaluations"],
            "Converged": "Yes" if r["converged"] else "No",
            "Wall Time (s)": f"{r['wall_time']:.1f}"
        })
    df_zero_tbl = pd.DataFrame(zero_table)
    md_output.append(to_markdown_table(df_zero_tbl))
    md_output.append("\n")

    # 2. Perturbed Initializations (Local Robustness around Chemical Reference)
    md_output.append("\n## 2. Perturbed Initializations (Combined $\\sigma \\in \\{0.01, 0.05, 0.1, 0.3\\}$, 40 runs per config)\n")
    md_output.append("*Evaluates optimizer basin stability when starting from chemically informed parameters with realistic perturbation noise.*\n\n")
    sub_pert = df[df["init_group"] == "perturbed"]
    pert_rows = []
    for (a_name, opt_name), group in sub_pert.groupby(["ansatz", "optimizer"]):
        errs = group["error_mHa"].values
        q25, med, q75 = np.percentile(errs, [25, 50, 75])
        ci_l, ci_u = compute_bootstrap_median_ci(errs)
        pert_rows.append({
            "Ansatz": a_name,
            "Optimizer": opt_name,
            "N": len(errs),
            "Median Error (mHa)": med,
            "IQR (mHa)": q75 - q25,
            "95% CI (mHa)": f"[{ci_l:.4f}, {ci_u:.4f}]",
            "Median % Corr": np.median(group["pct_corr_recovered"].values),
            "Success Rate": f"{np.mean(group['success'].values)*100.0:.1f}%",
            "Median Evals": int(np.median(group["total_evaluations"].values))
        })
    df_pert_tbl = pd.DataFrame(pert_rows)
    df_pert_tbl.sort_values(by="Median Error (mHa)", inplace=True)
    df_pert_tbl["Median Error (mHa)"] = df_pert_tbl["Median Error (mHa)"].apply(lambda x: f"{x:.4f}")
    df_pert_tbl["IQR (mHa)"] = df_pert_tbl["IQR (mHa)"].apply(lambda x: f"{x:.4f}")
    df_pert_tbl["Median % Corr"] = df_pert_tbl["Median % Corr"].apply(lambda x: f"{x:.1f}%")
    md_output.append(to_markdown_table(df_pert_tbl))
    md_output.append("\n")

    # 3. Random Initialization (Global Optimization Robustness from U(0, 2pi)^N)
    md_output.append("\n## 3. Random Initialization ($U(0, 2\\pi)^N$, 10 seeds)\n")
    md_output.append("*Evaluates global optimization ability and resistance to barren plateaus / local minima in the absence of chemical insight.*\n\n")
    sub_rand = df[df["init"] == "random"]
    rand_rows = []
    for (a_name, opt_name), group in sub_rand.groupby(["ansatz", "optimizer"]):
        errs = group["error_mHa"].values
        q25, med, q75 = np.percentile(errs, [25, 50, 75])
        ci_l, ci_u = compute_bootstrap_median_ci(errs)
        rand_rows.append({
            "Ansatz": a_name,
            "Optimizer": opt_name,
            "N": len(errs),
            "Median Error (mHa)": med,
            "IQR (mHa)": q75 - q25,
            "95% CI (mHa)": f"[{ci_l:.4f}, {ci_u:.4f}]",
            "Median % Corr": np.median(group["pct_corr_recovered"].values),
            "Success Rate": f"{np.mean(group['success'].values)*100.0:.1f}%",
            "Median Evals": int(np.median(group["total_evaluations"].values))
        })
    df_rand_tbl = pd.DataFrame(rand_rows)
    df_rand_tbl.sort_values(by="Median Error (mHa)", inplace=True)
    df_rand_tbl["Median Error (mHa)"] = df_rand_tbl["Median Error (mHa)"].apply(lambda x: f"{x:.4f}")
    df_rand_tbl["IQR (mHa)"] = df_rand_tbl["IQR (mHa)"].apply(lambda x: f"{x:.4f}")
    df_rand_tbl["Median % Corr"] = df_rand_tbl["Median % Corr"].apply(lambda x: f"{x:.1f}%")
    md_output.append(to_markdown_table(df_rand_tbl))
    md_output.append("\n")

    # Save markdown tables
    with open(TABLES_MD, "w", encoding="utf-8") as f:
        f.write("".join(md_output))
    print(f"Generated comprehensive ranking tables in {TABLES_MD}.")

    # Generate Figures
    generate_figures(df)


def generate_figures(df: pd.DataFrame):
    hf_gap = 8.1392
    chem_acc = 1.6000

    # 1. Error vs Evaluations per init type
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    init_types = ["zero", "perturbed_0.05", "random"]
    titles = ["Zero Initialization (Deterministic)", "Perturbed (sigma=0.05)", "Random U(0, 2pi)"]

    for ax, itype, title in zip(axes, init_types, titles):
        sub = df[df["init"] == itype]
        for a_name in sub["ansatz"].unique():
            a_sub = sub[sub["ansatz"] == a_name]
            if not a_sub.empty:
                med_eval = a_sub.groupby("optimizer")["total_evaluations"].median()
                med_err = a_sub.groupby("optimizer")["error_mHa"].median()
                ax.scatter(med_eval, med_err, label=a_name, s=60, alpha=0.85)

        ax.axhline(hf_gap, color="red", linestyle="--", linewidth=1.5, label="HF Gap (8.139 mHa)")
        ax.axhline(chem_acc, color="green", linestyle=":", linewidth=1.5, label="Chemical Acc (1.6 mHa)")
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Median Evaluations", fontsize=10)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)

    axes[0].set_ylabel("Error vs CASCI (mHa, log-scale)", fontsize=11)
    axes[2].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    fig_path = FIGURES_DIR / "hard_error_vs_evals.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Generated {fig_path}")

    # 2. Boxplots of Error per Optimizer
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, itype, title in zip(axes, init_types, titles):
        sub = df[df["init"] == itype]
        if not sub.empty:
            sub.boxplot(column="error_mHa", by="optimizer", ax=ax, grid=False)
        ax.axhline(hf_gap, color="red", linestyle="--", linewidth=1.2, label="HF Gap (8.139 mHa)")
        ax.axhline(chem_acc, color="green", linestyle=":", linewidth=1.2, label="Chemical Acc (1.6 mHa)")
        ax.set_yscale("log")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Optimizer", fontsize=10)

    axes[0].set_ylabel("Error vs CASCI (mHa, log-scale)", fontsize=11)
    fig.suptitle("Hard Benchmark VQE Error Distributions per Optimizer", fontsize=13, y=1.02)
    plt.tight_layout()
    fig_path2 = FIGURES_DIR / "hard_error_boxplots.png"
    plt.savefig(fig_path2, dpi=300)
    plt.close()
    print(f"Generated {fig_path2}")

    # 3. Heatmap of Median % Correlation Recovered
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for ax, itype, title in zip(axes, init_types, titles):
        sub = df[df["init"] == itype]
        if not sub.empty:
            pivot = sub.pivot_table(index="ansatz", columns="optimizer", values="pct_corr_recovered", aggfunc="median")
            im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            plt.colorbar(im, ax=ax, label="Median % Corr Recovered")
        ax.set_title(f"Median % Corr ({title})", fontsize=11, fontweight="bold")

    plt.tight_layout()
    fig_path3 = FIGURES_DIR / "hard_pct_corr_heatmap.png"
    plt.savefig(fig_path3, dpi=300)
    plt.close()
    print(f"Generated {fig_path3}")


if __name__ == "__main__":
    generate_phase_h_analysis()
