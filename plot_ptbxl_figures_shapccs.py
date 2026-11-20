"""PTB-XL paper figures with ShapCCS (B5) added as a sixth method.

Imports plot_ptbxl_figures.py unchanged and reuses its log parsing, curves, palette and axis style. The
original figures are left as they are; this script writes, into Multimodal_Federated_Learning/figures/:
  fig_ptbxl_{10,20,30,40}mb_shapccs.png       all methods at one budget (Figure: ptbxl_budgets)
  fig_ptbxl_budget_bars_shapccs.pdf           final accuracy, methods x budgets
  fig_ptbxl_budget_summary_zoom_shapccs.{pdf,png}  final accuracy vs budget, y-axis 50-70%
  ptbxl_final_values_shapccs.csv              every plotted final value (seed means)

ShapCCS runs: ShapCCS/torch_port/result_updated/ShapCCS_ptbxl_mm_p44_mm_*_C0.2_pf0.7_rem2_*_seed<s>.txt,
53/115/167/229 rounds for 10/20/30/40 MB; seeds 47 43 44 45 46 at 10 and 20 MB, 42-46 at 30 and 40 MB.
The final values equal the ShapCCS *_summary.txt files (Table 7).

Usage:
  venv/bin/python plot_ptbxl_figures_shapccs.py
"""
import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import plot_ptbxl_figures as base

BUDGETS = base.BUDGETS
MAJORITY = base.MAJORITY
SHAPCCS_ROUNDS = {10: 53, 20: 115, 30: 167, 40: 229}
SHAPCCS_SEEDS = {10: ("47", "43", "44", "45", "46"), 20: ("47", "43", "44", "45", "46"),
                 30: ("42", "43", "44", "45", "46"), 40: ("42", "43", "44", "45", "46")}
METHODS = base.METHODS + ["ShapCCS"]
COLORS = dict(base.COLORS, ShapCCS="#1a1a1a")  # black, as in the UCI HAR and mHealth figures
MARKERS = dict(base.MARKERS, ShapCCS="X")
YMIN, YMAX = 50, 70


def shapccs_logs(b):
    pat = os.path.join(base.HERE, "ShapCCS", "torch_port", "result_updated",
                       f"ShapCCS_{base.DS}_*MB_dirichlet*_it{SHAPCCS_ROUNDS[b]}_C0.2_pf0.7_rem2_*_seed*.txt")
    out = {}
    for f in glob.glob(pat):
        m = re.search(r"_seed(\d+)\.txt$", f)
        if m and m.group(1) in SHAPCCS_SEEDS[b]:
            out[m.group(1)] = f
    if len(out) != len(SHAPCCS_SEEDS[b]):
        raise SystemExit(f"ShapCCS {b} MB: found seeds {sorted(out)}, expected {SHAPCCS_SEEDS[b]}")
    return out


def shapccs_final(b, files):
    """Final accuracy mean/std from the full-precision npz (the logs round to 4 decimals), as summarize_seeds."""
    accs = []
    for s in files:
        p = os.path.join(base.HERE, "ShapCCS", "torch_port", "results",
                         f"ShapCCS_ptbxl_mm_p44_K10_C0.2_pf0.7_rem2_R{SHAPCCS_ROUNDS[b]}_seed{s}.npz")
        accs.append(float(np.load(p)["test_acc"][-1]) * 100)
    return float(np.mean(accs)), float(np.std(accs))


def method_logs(name, b):
    if name == "ShapCCS":
        return shapccs_logs(b)
    if name == "IC":  # the IC PTB-XL logs now sit in CommEfficient/result_updated/ptbxl/
        c, r = base.IC_RUN[b]
        return base.logs(os.path.join(base.HERE, "CommEfficient", "result_updated", "ptbxl",
                                      f"IC_{base.DS}_*MB_*_it{r}_r3xc{c}_*_seed*.txt"))
    return base.method_logs(name, b)


def config_label(name, b):
    if name == "ShapCCS":
        return f"2→1 of 10 clients · 70% of data pruned · {SHAPCCS_ROUNDS[b]} rds"
    if name == "Hybrid (ours)":
        c, k, r = base.HYB_MARKED[b]
        return f"{int(k * 10)}/10 clients · Top 1 · 3×{c} · {r} rds"
    if name == "IC":
        return f"10/10 clients · 3×{base.IC_RUN[b][0]} · {base.IC_RUN[b][1]} rds"
    if name == "MFedMC":
        return f"2/10 clients · Top 1 · uncompressed · {base.MFEDMC_ROUNDS[b]} rds"
    if name == "PFMH":
        return f"10/10 clients · 64-bit hash · {base.PFMH_ROUNDS[b]} rds"
    return f"10/10 clients · limb leads missing in 80% of batches · {base.FEDMOBILE_ROUNDS[b]} rds"


def budget_plot(b, out_dir, rows):
    """Same figure as plot_ptbxl_figures.budget_plot, with the ShapCCS curve."""
    fig, ax = plt.subplots(figsize=(10, 5.4))
    xmax = 0
    for name in METHODS:
        files = method_logs(name, b)
        if not files:
            print(f"  missing logs: {name} at {b} MB")
            continue
        x, y, mean, sd, n = base.curve(files)
        if name == "ShapCCS":
            mean, sd = shapccs_final(b, files)
        xmax = max(xmax, x[-1])
        few = len(x) <= 25
        cfg = config_label(name, b)
        ax.plot(x, y, color=COLORS[name], lw=2.6 if name.startswith("Hybrid") else 1.8,
                marker=MARKERS.get(name, "o") if few else None, ms=5,
                label=f"{name}: {mean:.2f}% at {x[-1]:.2f} MB  ({cfg})")
        rows.append((b, name, cfg, n, x[-1], mean, sd))
    base.style_axes(ax, b, xmax * 1.03)
    ax.set_title(f"PTB-XL, ~{b} MB upload budget: test accuracy vs cumulative upload (mean over seeds)", fontsize=11)
    fig.tight_layout()
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, fontsize=8.5, frameon=False)
    p = os.path.join(out_dir, f"fig_ptbxl_{b}mb_shapccs.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def bars_plot(final, out_dir):
    """plot_ptbxl_figures' grouped bars with a sixth bar per budget."""
    fig, ax = plt.subplots(figsize=(7, 3.8))
    n = len(METHODS)
    w = 0.8 / n
    for i, name in enumerate(METHODS):
        xs = np.arange(len(BUDGETS)) + (i - (n - 1) / 2) * w
        ax.bar(xs, [final[(b, name)] for b in BUDGETS], w * 0.92, color=COLORS[name], label=name)
    ax.axhline(MAJORITY, color="0.4", ls="--", lw=1, label=f"Majority class ({MAJORITY:.1f}%)")
    ax.set_xticks(np.arange(len(BUDGETS)))
    ax.set_xticklabels([f"{b} MB" for b in BUDGETS])
    ax.set_ylim(0, 80)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_xlabel("Cumulative upload budget")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=4, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, 1.2), frameon=False)
    fig.tight_layout()
    p = os.path.join(out_dir, "fig_ptbxl_budget_bars_shapccs.pdf")
    fig.savefig(p)
    plt.close(fig)
    print("wrote", p)


def zoom_plot(final, out_dir):
    """plot_ptbxl_budget_summary_zoom's figure (y 50-70%, MFedMC as arrows below the axis) plus ShapCCS."""
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for name in METHODS:
        ys = [final[(b, name)] for b in BUDGETS]
        if all(y >= YMIN for y in ys):
            ax.plot(BUDGETS, ys, color=COLORS[name], marker=MARKERS.get(name, "o"),
                    lw=2 if name.startswith("Hybrid") else 1.5, label=name)
            continue
        ax.plot(BUDGETS, [YMIN + 0.6] * len(BUDGETS), ls="none", marker="v", ms=7,
                color=COLORS[name], label=f"{name} (below axis)", clip_on=False)
        for b, y in zip(BUDGETS, ys):
            ax.annotate(f"{y:.1f}%", (b, YMIN + 0.6), xytext=(0, 7), textcoords="offset points",
                        ha="center", fontsize=7.5, color=COLORS[name])
    ax.axhline(MAJORITY, color="0.4", ls="--", lw=1, label=f"Majority class ({MAJORITY:.1f}%)")
    ax.set_xticks(BUDGETS)
    ax.set_ylim(YMIN, YMAX)
    ax.set_yticks(range(YMIN, YMAX + 1, 5))
    ax.set_xlabel("Cumulative upload budget (MB)")
    ax.set_ylabel("Final test accuracy (%)")
    ax.grid(alpha=0.3)
    ax.legend(ncol=4, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, 1.2), frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = os.path.join(out_dir, f"fig_ptbxl_budget_summary_zoom_shapccs.{ext}")
        fig.savefig(p, dpi=200)
        print("wrote", p)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", default=os.path.join(base.HERE, "Multimodal_Federated_Learning", "figures"))
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for b in BUDGETS:
        budget_plot(b, args.out_dir, rows)
    final = {(b, n): m for b, n, _, _, _, m, _ in rows}
    bars_plot(final, args.out_dir)
    zoom_plot(final, args.out_dir)
    csv = os.path.join(args.out_dir, "ptbxl_final_values_shapccs.csv")
    with open(csv, "w", encoding="utf-8") as f:
        f.write("budget_MB,method,config,seeds,final_cum_up_MB,final_acc_mean,final_acc_std\n")
        for b, n, cfg, k, x, m, sd in rows:
            f.write(f"{b},{n},{cfg},{k},{x:.2f},{m:.2f},{sd:.2f}\n")
    print("wrote", csv)


if __name__ == "__main__":
    main()
