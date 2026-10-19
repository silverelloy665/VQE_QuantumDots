"""
Regenerates and updates results.xlsx with verified Round 2 data from:
- data/hard_benchmark_summary.csv
- data/active_space_scan.csv
- data/circuit_costs_hard.csv
- data/reconciled_hardware_jobs.csv
- data/hardware_job_spec_v2.json

Strictly adheres to the rule: No fabricated, estimated, or hard-coded result numbers.
"""
import json
from pathlib import Path
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
EXCEL_PATH = REPO_ROOT / "results.xlsx"


def format_worksheet(ws, title_color="1F497D"):
    """Applies clean engineering styling to an Excel worksheet."""
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color=title_color, end_color=title_color, fill_type="solid")
    regular_font = Font(name="Calibri", size=10)
    thin_border = Border(
        left=Side(style="thin", color="D9D9D9"),
        right=Side(style="thin", color="D9D9D9"),
        top=Side(style="thin", color="D9D9D9"),
        bottom=Side(style="thin", color="D9D9D9")
    )

    for cell in ws[1]:
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.font = regular_font
            cell.border = thin_border
            if isinstance(cell.value, float):
                if abs(cell.value) >= 1000:
                    cell.number_format = "#,##0.00"
                elif abs(cell.value) < 1e-4 and cell.value != 0:
                    cell.number_format = "0.0000E+00"
                else:
                    cell.number_format = "0.0000"

    # Auto-adjust column widths
    for col in ws.columns:
        max_len = max(len(str(cell.value or '')) for cell in col)
        col_letter = openpyxl.utils.get_column_letter(col[0].column)
        ws.column_dimensions[col_letter].width = min(max(max_len + 3, 12), 45)


def update_results_excel():
    print(f"Loading {EXCEL_PATH}...")
    wb = openpyxl.load_workbook(EXCEL_PATH) if EXCEL_PATH.exists() else openpyxl.Workbook()

    # 1. Hard Benchmark Summary
    summary_csv = DATA_DIR / "hard_benchmark_summary.csv"
    if summary_csv.exists():
        df_summary = pd.read_csv(summary_csv)
        sheet_name = "Round 2 - Hard Benchmark"
        if sheet_name in wb.sheetnames:
            del wb[sheet_name]
        ws = wb.create_sheet(title=sheet_name)
        for r in dataframe_to_rows(df_summary, index=False, header=True):
            ws.append(r)
        format_worksheet(ws, title_color="1F4E79")
        print(f"  Added sheet '{sheet_name}' ({len(df_summary)} rows)")

    # 2. Active Space Scan
    scan_csv = DATA_DIR / "active_space_scan.csv"
    if scan_csv.exists():
        df_scan = pd.read_csv(scan_csv)
        sheet_name = "Round 2 - Active Space Scan"
        if sheet_name in wb.sheetnames:
            del wb[sheet_name]
        ws = wb.create_sheet(title=sheet_name)
        for r in dataframe_to_rows(df_scan, index=False, header=True):
            ws.append(r)
        format_worksheet(ws, title_color="203764")
        print(f"  Added sheet '{sheet_name}' ({len(df_scan)} rows)")

    # 3. Circuit Costs
    costs_csv = DATA_DIR / "circuit_costs_hard.csv"
    if costs_csv.exists():
        df_costs = pd.read_csv(costs_csv)
        sheet_name = "Round 2 - Circuit Costs"
        if sheet_name in wb.sheetnames:
            del wb[sheet_name]
        ws = wb.create_sheet(title=sheet_name)
        for r in dataframe_to_rows(df_costs, index=False, header=True):
            ws.append(r)
        format_worksheet(ws, title_color="2F5597")
        print(f"  Added sheet '{sheet_name}' ({len(df_costs)} rows)")

    # 4. Reconciled Hardware Jobs
    jobs_csv = DATA_DIR / "reconciled_hardware_jobs.csv"
    if jobs_csv.exists():
        df_jobs = pd.read_csv(jobs_csv)
        sheet_name = "Round 2 - Reconciled HW Jobs"
        if sheet_name in wb.sheetnames:
            del wb[sheet_name]
        ws = wb.create_sheet(title=sheet_name)
        for r in dataframe_to_rows(df_jobs, index=False, header=True):
            ws.append(r)
        format_worksheet(ws, title_color="333f48")
        print(f"  Added sheet '{sheet_name}' ({len(df_jobs)} rows)")

    # 5. Hardware Job Spec V2
    spec_v2_json = DATA_DIR / "hardware_job_spec_v2.json"
    if spec_v2_json.exists():
        with open(spec_v2_json, "r", encoding="utf-8") as f:
            specs = json.load(f)
        spec_rows = []
        for s in specs:
            spec_rows.append({
                "Job_ID": s["job_id"],
                "Priority": s["priority"],
                "Experiment": s["experiment_name"],
                "Backend": s["backend"],
                "Shots": s["shots"],
                "Resilience": s["resilience_type"],
                "QWC_Circuits": s["qwc_circuits"],
                "2Q_Gates": s["transpiled_2q_gates"],
                "Depth": s["transpiled_depth"],
                "Est_Fidelity": s["estimated_fidelity"],
                "Noiseless_E_Ha": s["expected_noiseless_energy_Ha"],
                "CASCI_E_Ha": s["casci_energy_Ha"],
                "Error_mHa": s["error_vs_casci_mHa"],
                "Est_QPU_Time_s": s["estimated_qpu_time_seconds"],
                "Recommendation": s["recommendation"]
            })
        df_specs = pd.DataFrame(spec_rows)
        sheet_name = "Round 2 - Hardware Jobs V2"
        if sheet_name in wb.sheetnames:
            del wb[sheet_name]
        ws = wb.create_sheet(title=sheet_name)
        for r in dataframe_to_rows(df_specs, index=False, header=True):
            ws.append(r)
        format_worksheet(ws, title_color="203764")
        print(f"  Added sheet '{sheet_name}' ({len(df_specs)} rows)")

    # Save
    wb.save(EXCEL_PATH)
    print(f"Successfully updated and saved {EXCEL_PATH}.")


if __name__ == "__main__":
    update_results_excel()

