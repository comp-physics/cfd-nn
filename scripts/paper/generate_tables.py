#!/usr/bin/env python3
"""Generate paper tables in LaTeX (booktabs) and Markdown formats.

Usage:
    python scripts/paper/generate_tables.py

Outputs to results/paper/tables/ (created if needed).
No dependencies beyond the Python 3 standard library.
"""

import os
import pathlib

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "paper" / "tables"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _latex_escape(s):
    """Escape special LaTeX characters in a string."""
    s = str(s)
    # Don't escape things already containing LaTeX commands
    if "\\" in s or "{" in s:
        return s
    s = s.replace("&", r"\&")
    s = s.replace("%", r"\%")
    s = s.replace("#", r"\#")
    s = s.replace("~", r"\textasciitilde{}")
    return s


def _md_row(cells):
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _md_sep(ncols):
    return "|" + "|".join("---" for _ in range(ncols)) + "|"


def generate_latex_table(caption, label, headers, rows, notes=None,
                         col_align=None):
    """Return a complete LaTeX table string using booktabs."""
    ncols = len(headers)
    if col_align is None:
        col_align = "l" + "r" * (ncols - 1)

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption + "}")
    lines.append(r"\label{" + label + "}")
    lines.append(r"\begin{tabular}{" + col_align + "}")
    lines.append(r"\toprule")
    lines.append(" & ".join(r"\textbf{" + _latex_escape(h) + "}" for h in headers) + r" \\")
    lines.append(r"\midrule")
    for row in rows:
        lines.append(" & ".join(_latex_escape(str(c)) for c in row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if notes:
        lines.append(r"\vspace{2pt}")
        lines.append(r"\begin{minipage}{\linewidth}")
        lines.append(r"\footnotesize")
        for note in notes:
            lines.append(r"\textit{" + _latex_escape(note) + r"}")
            lines.append(r"\\")
        lines.append(r"\end{minipage}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_md_table(title, headers, rows, notes=None):
    """Return a Markdown table string."""
    lines = [f"## {title}", ""]
    lines.append(_md_row(headers))
    lines.append(_md_sep(len(headers)))
    for row in rows:
        lines.append(_md_row(row))
    if notes:
        lines.append("")
        for note in notes:
            lines.append(f"*{note}*")
    lines.append("")
    return "\n".join(lines)


def write_and_print(name, latex_str, md_str):
    """Write .tex and .md files and print both to stdout."""
    tex_path = OUTPUT_DIR / f"{name}.tex"
    md_path = OUTPUT_DIR / f"{name}.md"
    tex_path.write_text(latex_str + "\n")
    md_path.write_text(md_str + "\n")
    print(f"{'=' * 72}")
    print(f"  {name}")
    print(f"{'=' * 72}")
    print()
    print(md_str)
    print()
    print(latex_str)
    print()


# ---------------------------------------------------------------------------
# Table 1: Model Summary
# ---------------------------------------------------------------------------

def table1_model_summary():
    headers = ["Model", "Architecture", "Parameters", "FLOPs/cell",
               "Weight Size", "Deployable"]
    rows = [
        ["Baseline (mixing length)", "Algebraic", "0", "~2", "0", "Yes"],
        ["k-omega", "2-eq transport", "0", "~50", "0", "Yes"],
        ["SST k-omega", "2-eq transport", "0", "~80", "0", "Yes"],
        ["GEP", "Algebraic", "0", "~40", "0", "Yes"],
        ["EARSM-WJ", "SST + tensor algebra", "0", "~120", "0", "Yes"],
        ["EARSM-GS", "SST + tensor algebra", "0", "~120", "0", "Yes"],
        ["EARSM-Pope", "SST + tensor algebra", "0", "~120", "0", "Yes"],
        ["MLP", r"5$\to$32$\to$32$\to$1 (Tanh)", "1,249", "1,249", "56 KB", "Yes"],
        ["MLP-Large", r"5$\to$128$^4\to$1 (Tanh)", "50,049", "50,049", "896 KB", "Yes"],
        ["TBNN", r"5$\to$64$^3\to$10 (Tanh)", "9,354", "10,254", "196 KB", "Yes"],
        ["PI-TBNN", r"5$\to$64$^3\to$10 (Tanh)", "9,354", "10,254", "196 KB", "Yes"],
        ["TBRF (1 tree)", r"10 forests $\times$ 1 tree", "282,902 nodes", "~2,200", "5.7 MB", "Experimental"],
        ["TBRF (5 trees)", r"10 forests $\times$ 5 trees", "1,432,612 nodes", "~10,200", "29 MB", "Experimental"],
        ["TBRF (10 trees)", r"10 forests $\times$ 10 trees", "2,797,130 nodes", "~20,200", "56 MB", "Experimental"],
        ["TBRF (200 trees)", r"10 forests $\times$ 200 trees", "55,051,026 nodes", "~400,200", "3.3 GB", "No"],
    ]
    # For markdown, replace LaTeX math with plain text
    md_rows = []
    for row in rows:
        md_row = list(row)
        md_row[1] = (md_row[1]
                     .replace(r"$\to$", "->")
                     .replace(r"$\times$", "x")
                     .replace(r"$^4", "^4")
                     .replace(r"$^3", "^3")
                     .replace(r"\to$", "->")
                     .replace(r"$", "")
                     .replace("  ", " "))
        md_rows.append(md_row)

    notes = [
        "FLOPs/cell for NN includes feature computation (~900 FLOPs for invariants + tensor basis for TBNN/TBRF).",
        "TBRF FLOPs estimated as: 900 (features) + n_coefficients x n_trees x avg_depth x 2 (comparison + branch).",
        "For MLP, FLOPs = weights count (each weight = 1 multiply + 1 add) + activations.",
    ]

    latex = generate_latex_table(
        caption="Summary of turbulence closure models.",
        label="tab:model_summary",
        headers=headers,
        rows=rows,
        notes=notes,
        col_align="llrrrr",
    )
    md = generate_md_table("Table 1: Model Summary", headers, md_rows, notes)
    write_and_print("table1_model_summary", latex, md)


# ---------------------------------------------------------------------------
# Table 4a: GPU Profiling -- Cylinder
# ---------------------------------------------------------------------------

def table4a_gpu_profiling_cylinder():
    headers = ["Model", "IBM (s)", "Poisson (s)", "Turb Update (s)",
               "Turb Transport (s)", "Total (s)", "Overhead"]
    rows = [
        ["Baseline",  "---", "11.4", "2.33", "---", "45.0", "---"],
        ["k-omega",   "2.65", "11.8", "0.47", "0.85", "45.1", "+0.2%"],
        ["SST",       "2.66", "11.4", "1.26", "2.22", "46.9", "+4.2%"],
        ["GEP",       "2.66", "11.3", "1.18", "---",  "46.3", "+2.9%"],
        ["EARSM-WJ",  "2.67", "11.3", "1.63", "2.22", "47.4", "+5.3%"],
        ["EARSM-GS",  "2.67", "11.3", "1.58", "2.22", "47.3", "+5.1%"],
        ["EARSM-Pope", "2.67", "11.3", "1.46", "2.22", "47.1", "+4.7%"],
        ["NN-MLP",    "2.65", "11.3", "12.78", "---", "55.6", "+23.6%"],
        ["NN-TBNN",   "2.67", "11.4", "85.55", "---", "129.0", "+186.7%"],
    ]

    latex = generate_latex_table(
        caption="GPU profiling: Cylinder Re=100, L40S, 50K steps.",
        label="tab:gpu_profiling_cylinder",
        headers=headers,
        rows=rows,
        col_align="lrrrrrr",
    )
    md = generate_md_table(
        "Table 4a: GPU Profiling (Cylinder Re=100, L40S, 50K steps)",
        headers, rows,
    )
    write_and_print("table4a_gpu_profiling_cylinder", latex, md)


# ---------------------------------------------------------------------------
# Table 4b: GPU Profiling -- Airfoil
# ---------------------------------------------------------------------------

def table4b_gpu_profiling_airfoil():
    headers = ["Model", "IBM (s)", "Poisson (s)", "Turb Update (s)",
               "Turb Transport (s)", "Total (s)", "Overhead"]

    # Compute overhead vs baseline (83.0 s)
    base_total = 83.0
    raw = [
        ["Baseline",  "---",  "26.6", "3.01", "---",  83.0],
        ["k-omega",   "6.99", "26.6", "0.58", "2.03", 86.0],
        ["SST",       "7.08", "26.6", "2.14", "4.63", 90.0],
        ["GEP",       "7.08", "26.6", "1.89", "---",  88.5],
        ["EARSM-WJ",  "7.15", "26.6", "3.65", "4.59", 91.7],
        ["EARSM-GS",  "7.15", "26.6", "3.30", "4.65", 91.4],
        ["EARSM-Pope", "7.12", "26.6", "2.80", "4.63", 90.8],
        ["NN-MLP",    "7.18", "26.6", "40.77", "---", 124.3],
        ["NN-TBNN",   "7.20", "26.7", "543.91", "---", 628.4],
    ]
    rows = []
    for r in raw:
        total = r[5]
        if r[0] == "Baseline":
            overhead = "---"
        else:
            overhead = f"+{(total - base_total) / base_total * 100:.1f}%"
        rows.append(r[:5] + [str(total), overhead])

    latex = generate_latex_table(
        caption="GPU profiling: Airfoil, L40S, 50K steps.",
        label="tab:gpu_profiling_airfoil",
        headers=headers,
        rows=rows,
        col_align="lrrrrrr",
    )
    md = generate_md_table(
        "Table 4b: GPU Profiling (Airfoil, L40S, 50K steps)",
        headers, rows,
    )
    write_and_print("table4b_gpu_profiling_airfoil", latex, md)


# ---------------------------------------------------------------------------
# Table 8: TBRF Tree Count Sweep
# ---------------------------------------------------------------------------

def table8_tbrf_sweep():
    headers = ["Trees/coeff", "Total Nodes", "Binary Size",
               "Val RMSE(b)", "vs TBNN"]
    rows = [
        ["1",   "282,902",    "5.7 MB",    "0.0778", "-7.9%"],
        ["5",   "1,432,612",  "28.7 MB",   "0.0678", "-19.8%"],
        ["10",  "2,797,130",  "55.9 MB",   "0.0650", "-23.1%"],
        ["20",  "5,698,348",  "114.0 MB",  "0.0651", "-23.0%"],
        ["50",  "13,958,032", "279.2 MB",  "0.0642", "-24.0%"],
        ["100", "27,859,152", "557.2 MB",  "0.0639", "-24.4%"],
        ["200", "55,051,026", "1,101 MB",  "0.0637", "-24.6%"],
    ]

    latex = generate_latex_table(
        caption="TBRF tree count sweep: accuracy vs.~model size.",
        label="tab:tbrf_sweep",
        headers=headers,
        rows=rows,
        col_align="rrrrr",
    )
    md = generate_md_table("Table 8: TBRF Tree Count Sweep", headers, rows)
    write_and_print("table8_tbrf_sweep", latex, md)


# ---------------------------------------------------------------------------
# Table 9: PI-TBNN Beta Sweep
# ---------------------------------------------------------------------------

def table9_pi_tbnn_sweep():
    headers = ["Beta", "L2 Reg (alpha)", "Val RMSE(b)", "Epochs",
               "vs TBNN (0.0845)"]
    rows = [
        ["0 (TBNN)", "0",    "0.0845", "694", "baseline"],
        ["0.001",    "1e-6", "0.0852", "729", "+0.8%"],
        ["0.01",     "1e-6", "0.0909", "676", "+7.6%"],
        ["0.1 (L2 bug)", "0.01", "0.1215", "152", "+43.8%"],
        ["1.0 (L2 bug)", "0.01", "0.1203", "151", "+42.4%"],
    ]
    notes = [
        "The L2 bug (alpha=0.01) caused ~825x larger regularization than MSE, dominating the loss.",
    ]

    latex = generate_latex_table(
        caption="PI-TBNN physics-informed loss weight sweep.",
        label="tab:pi_tbnn_sweep",
        headers=headers,
        rows=rows,
        notes=notes,
        col_align="llrrr",
    )
    md = generate_md_table("Table 9: PI-TBNN Beta Sweep", headers, rows, notes)
    write_and_print("table9_pi_tbnn_sweep", latex, md)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}\n")

    table1_model_summary()
    table4a_gpu_profiling_cylinder()
    table4b_gpu_profiling_airfoil()
    table8_tbrf_sweep()
    table9_pi_tbnn_sweep()

    print(f"{'=' * 72}")
    print(f"All tables written to {OUTPUT_DIR}")
    print(f"  .tex files: LaTeX (booktabs)")
    print(f"  .md  files: Markdown")


if __name__ == "__main__":
    main()
