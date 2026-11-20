"""Paper figures for PTB-XL (2 modalities, partition seed 44): every method's
seed-mean test accuracy against cumulative upload, read straight from the
seed-tagged text logs.

Writes, into Multimodal_Federated_Learning/figures/ by default:
  fig_ptbxl_{10,20,30,40}mb.png   all methods at one budget; Hybrid = the
                                  configuration marked in the Results sheet
  fig_ptbxl_cfg_{10,20,30,40}mb.png  the nine Hybrid configurations of a budget
  fig_ptbxl_budget_bars.pdf       final accuracy, methods x budgets
  fig_ptbxl_budget_summary.pdf    final accuracy against the budget
  ptbxl_final_values.csv          every plotted final value (seed means)

The final value of a curve is the mean over seeds of the last round's
test_acc, i.e. the same number as the *_summary.txt files and the sheet.
Same palette and method order as plot_mhealth_budget_comparison.py.

Usage:
  venv/bin/python plot_ptbxl_figures.py
"""
import argparse
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "HybridSketchMFedMC_v2"))
from plot_seed_mean import table  # noqa: E402  (same parser every method's plot script uses)

BUDGETS = (10, 20, 30, 40)
SEEDS = ("42", "43", "44", "45", "46")
MAJORITY = 1814 / 3220 * 100  # NORM share of the PTB-XL test set
DS = "ptbxl_mm_p44_mm"
HYB_DIRS = {10: "10MB", 20: "20MB", 30: "30 MB", 40: "40 MB"}

# Baseline runs per budget (the budget sweeps that exist) and the Hybrid
# configuration marked in the Results sheet: (sketch cols, client ratio, rounds).
IC_RUN = {10: (2000, 42), 20: (3000, 56), 30: (4000, 63), 40: (5000, 68)}
MFEDMC_ROUNDS = {10: 6, 20: 11, 30: 17, 40: 22}
PFMH_ROUNDS = {10: 3, 20: 6, 30: 8, 40: 11}
FEDMOBILE_ROUNDS = {10: 5, 20: 9, 30: 13, 40: 17}
HYB_MARKED = {10: (2000, 0.5, 84), 20: (3000, 0.5, 112), 30: (4000, 0.5, 125), 40: (5000, 0.3, 225)}

METHODS = ["Hybrid (ours)", "IC", "MFedMC", "PFMH", "FedMobile"]
COLORS = {"Hybrid (ours)": "#2a78d6", "IC": "#eb6834", "MFedMC": "#1baf7a",
          "PFMH": "#eda100", "FedMobile": "#6250d6"}
MARKERS = {"FedMobile": "s"}


def logs(pattern):
    """{seed: path} for the seed-tagged logs matching a glob (NAMB runs skipped)."""
    out = {}
    for f in glob.glob(pattern):
        m = re.search(r"_seed(\d+)\.txt$", f)
        if m and "NAMB" not in f and m.group(1) in SEEDS:
            out.setdefault(m.group(1), f)
    return out


def hybrid_pattern(b, cols, client, rounds):
    d = os.path.join(HERE, "HybridSketchMFedMC_v2", "result_updated", HYB_DIRS[b], "ptbxl")
    pat = f"HybridV2_{DS}_*MB_*_it{rounds}_r3xc{cols}_*_Client_{client}_*_seed*.txt"
    if not glob.glob(os.path.join(d, pat)):  # the 10 MB 3x1000 / 50% run sits in result_updated/ itself
        return os.path.join(HERE, "HybridSketchMFedMC_v2", "result_updated", pat)
    return os.path.join(d, pat)


def method_logs(name, b):
    if name == "Hybrid (ours)":
        return logs(hybrid_pattern(b, *HYB_MARKED[b]))
    if name == "IC":
        c, r = IC_RUN[b]
        return logs(os.path.join(HERE, "CommEfficient", "result_updated",
                                 f"IC_{DS}_*MB_*_it{r}_r3xc{c}_*_seed*.txt"))
    if name == "MFedMC":
        return logs(os.path.join(HERE, "MFedMC", "ptbxl", "result_updated",
                                 f"MFedMC_v2_{DS}_*MB_*_it{MFEDMC_ROUNDS[b]}_*_seed*.txt"))
    if name == "PFMH":
        return logs(os.path.join(HERE, "PMFH", "result_updated",
                                 f"PMFHfh_{DS}_*MB_dirichlet*_it{PFMH_ROUNDS[b]}_*_seed*.txt"))
    if name == "FedMobile":
        return logs(os.path.join(HERE, "FedMobile", "torch_port", "result_updated",
                                 f"FedMobile_{DS}_*MB_dirichlet*_it{FEDMOBILE_ROUNDS[b]}_*_seed*.txt"))
    raise ValueError(name)


def curve(files):
    """Seed-mean (cum_up_MB, test_acc %) over the rounds all seeds share."""
    tabs = [table(f) for f in files.values()]
    tabs = [t for t in tabs if t is not None]
    n = min(len(t["iter"]) for t in tabs)
    x = np.mean([t["cum_up_MB"][:n] for t in tabs], axis=0)
    y = np.mean([t["test_acc"][:n] * 100 for t in tabs], axis=0)
    finals = [t["test_acc"][-1] * 100 for t in tabs]
    return x, y, float(np.mean(finals)), float(np.std(finals)), len(tabs)


def hybrid_configs(b):
    """Every Hybrid configuration of a budget: {(cols, client, rounds): {seed: path}}."""
    d = os.path.join(HERE, "HybridSketchMFedMC_v2", "result_updated", HYB_DIRS[b], "ptbxl")
    files = glob.glob(os.path.join(d, f"HybridV2_{DS}_*_seed*.txt"))
    if b == 10:
        files += glob.glob(os.path.join(HERE, "HybridSketchMFedMC_v2", "result_updated",
                                        f"HybridV2_{DS}_*_seed*.txt"))
    out = {}
    for f in files:
        m = re.search(r"_it(\d+)_r3xc(\d+)_.*_Client_([\d.]+)_.*_seed(\d+)\.txt$", f)
        if not m or "NAMB" in f or m.group(4) not in SEEDS:
            continue
        out.setdefault((int(m.group(2)), float(m.group(3)), int(m.group(1))), {})[m.group(4)] = f
    return out


def style_axes(ax, b, xmax):
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.set_xlim(0, xmax)
    ax.grid(alpha=0.3)
    ax.axvline(b, color="0.5", ls=":", lw=1.2)
    ax.text(b, 96, f" {b} MB budget", color="0.4", fontsize=9, va="top")
    ax.axhline(MAJORITY, color="0.45", ls="--", lw=1, label=f"Majority class (always NORM): {MAJORITY:.2f}%")
    ax.set_xlabel("Cumulative upload (MB)")
    ax.set_ylabel("Test accuracy (%)")


def budget_plot(b, out_dir, rows):
    fig, ax = plt.subplots(figsize=(10, 5.4))
    xmax = 0
    for name in METHODS:
        files = method_logs(name, b)
        if not files:
            print(f"  missing logs: {name} at {b} MB")
            continue
        x, y, mean, sd, n = curve(files)
        xmax = max(xmax, x[-1])
        few = len(x) <= 25
        if name == "Hybrid (ours)":
            c, k, r = HYB_MARKED[b]
            cfg = f"{int(k * 10)}/10 clients · Top 1 · 3×{c} · {r} rds"
        elif name == "IC":
            cfg = f"10/10 clients · 3×{IC_RUN[b][0]} · {IC_RUN[b][1]} rds"
        elif name == "MFedMC":
            cfg = f"2/10 clients · Top 1 · uncompressed · {MFEDMC_ROUNDS[b]} rds"
        elif name == "PFMH":
            cfg = f"10/10 clients · 64-bit hash · {PFMH_ROUNDS[b]} rds"
        else:
            cfg = f"10/10 clients · limb leads missing in 80% of batches · {FEDMOBILE_ROUNDS[b]} rds"
        ax.plot(x, y, color=COLORS[name], lw=2.6 if name.startswith("Hybrid") else 1.8,
                marker=MARKERS.get(name, "o") if few else None, ms=5,
                label=f"{name}: {mean:.2f}% at {x[-1]:.2f} MB  ({cfg})")
        rows.append((b, name, cfg, n, x[-1], mean, sd))
    style_axes(ax, b, xmax * 1.03)
    ax.set_title(f"PTB-XL, ~{b} MB upload budget: test accuracy vs cumulative upload (mean over seeds)", fontsize=11)
    fig.tight_layout()
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, fontsize=8.5, frameon=False)
    p = os.path.join(out_dir, f"fig_ptbxl_{b}mb.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def config_plot(b, out_dir, rows):
    cfgs = hybrid_configs(b)
    widths = sorted({c for c, _, _ in cfgs})
    ratios = sorted({k for _, k, _ in cfgs})
    ccol = dict(zip(ratios, ["#2a78d6", "#eb6834", "#1baf7a"]))
    cstyle = dict(zip(widths, ["-", "--", ":"]))
    fig, ax = plt.subplots(figsize=(10, 5.4))
    lines, xmax = [], 0
    for (c, k, r), files in cfgs.items():
        x, y, mean, sd, n = curve(files)
        xmax = max(xmax, x[-1])
        marked = (c, k, r) == HYB_MARKED[b]
        (ln,) = ax.plot(x, y, color=ccol[k], ls=cstyle[c], lw=2.6 if marked else 1.6)
        lab = f"{int(k * 10)}/10 clients · 3×{c} · {r} rds: {mean:.2f}%" + (" (marked)" if marked else "") \
            + (f" [{n} seeds]" if n < len(SEEDS) else "")
        lines.append((mean, ln, lab))
        rows.append((b, "Hybrid config", f"{int(k * 10)}/10 · 3x{c} · {r} rds", n, x[-1], mean, sd))
    lines.sort(key=lambda t: -t[0])
    style_axes(ax, b, xmax * 1.03)
    ax.set_title(f"PTB-XL, ~{b} MB upload budget: Hybrid configurations (mean over seeds)", fontsize=11)
    maj = Line2D([], [], color="0.45", ls="--", lw=1)
    ax.legend([l for _, l, _ in lines] + [maj], [s for _, _, s in lines] + [f"Majority class (always NORM): {MAJORITY:.2f}%"],
              loc="lower right", fontsize=8.5, framealpha=0.95)
    fig.tight_layout()
    p = os.path.join(out_dir, f"fig_ptbxl_cfg_{b}mb.png")
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print("wrote", p)


def summary_plots(rows, out_dir):
    final = {(b, n): m for b, n, _, _, _, m, _ in rows if n in METHODS}
    # grouped bars
    fig, ax = plt.subplots(figsize=(7, 3.8))
    w = 0.16
    for i, name in enumerate(METHODS):
        xs = np.arange(len(BUDGETS)) + (i - 2) * w
        vals = [final[(b, name)] for b in BUDGETS]
        ax.bar(xs, vals, w * 0.92, color=COLORS[name], label=name)
    ax.axhline(MAJORITY, color="0.4", ls="--", lw=1, label=f"Majority class ({MAJORITY:.1f}%)")
    ax.set_xticks(np.arange(len(BUDGETS)))
    ax.set_xticklabels([f"{b} MB" for b in BUDGETS])
    ax.set_ylim(0, 80)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_xlabel("Cumulative upload budget")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=3, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, 1.2), frameon=False)
    fig.tight_layout()
    p = os.path.join(out_dir, "fig_ptbxl_budget_bars.pdf")
    fig.savefig(p)
    plt.close(fig)
    print("wrote", p)
    # final accuracy against the budget
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for name in METHODS:
        ax.plot(BUDGETS, [final[(b, name)] for b in BUDGETS], color=COLORS[name],
                marker=MARKERS.get(name, "o"), lw=2 if name.startswith("Hybrid") else 1.5, label=name)
    ax.axhline(MAJORITY, color="0.4", ls="--", lw=1, label=f"Majority class ({MAJORITY:.1f}%)")
    ax.set_xticks(BUDGETS)
    ax.set_ylim(0, 80)
    ax.set_xlabel("Cumulative upload budget (MB)")
    ax.set_ylabel("Final test accuracy (%)")
    ax.grid(alpha=0.3)
    ax.legend(ncol=3, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, 1.2), frameon=False)
    fig.tight_layout()
    p = os.path.join(out_dir, "fig_ptbxl_budget_summary.pdf")
    fig.savefig(p)
    plt.close(fig)
    print("wrote", p)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", default=os.path.join(HERE, "Multimodal_Federated_Learning", "figures"))
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for b in BUDGETS:
        budget_plot(b, args.out_dir, rows)
        config_plot(b, args.out_dir, rows)
    summary_plots(rows, args.out_dir)
    csv = os.path.join(args.out_dir, "ptbxl_final_values.csv")
    with open(csv, "w", encoding="utf-8") as f:
        f.write("budget_MB,method,config,seeds,final_cum_up_MB,final_acc_mean,final_acc_std\n")
        for b, n, cfg, k, x, m, sd in rows:
            f.write(f"{b},{n},{cfg},{k},{x:.2f},{m:.2f},{sd:.2f}\n")
    print("wrote", csv)


if __name__ == "__main__":
    main()
