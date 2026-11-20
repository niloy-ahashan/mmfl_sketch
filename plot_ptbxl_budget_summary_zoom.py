"""Zoomed version of fig_ptbxl_budget_summary.pdf: final accuracy against the
budget on PTB-XL with the y-axis cut to 50-70%, so the few-point differences
between methods are visible.

Reads the seed means that plot_ptbxl_figures.py wrote to
ptbxl_final_values.csv (it does not re-parse the logs), with the same palette,
markers and method order. MFedMC (15-29%) is below the axis: it is drawn as
down-arrows on the bottom edge, labelled with its values.

Writes fig_ptbxl_budget_summary_zoom.{pdf,png} next to the CSV; the original
figure is left as it is.

Usage:
  venv/bin/python plot_ptbxl_budget_summary_zoom.py
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

BUDGETS = (10, 20, 30, 40)
MAJORITY = 1814 / 3220 * 100  # NORM share of the PTB-XL test set
METHODS = ["Hybrid (ours)", "IC", "MFedMC", "PFMH", "FedMobile"]
COLORS = {"Hybrid (ours)": "#2a78d6", "IC": "#eb6834", "MFedMC": "#1baf7a",
          "PFMH": "#eda100", "FedMobile": "#6250d6"}
MARKERS = {"FedMobile": "s"}
YMIN, YMAX = 50, 70


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fig_dir", default=os.path.join(HERE, "Multimodal_Federated_Learning", "figures"))
    args = p.parse_args()

    final = {}
    with open(os.path.join(args.fig_dir, "ptbxl_final_values.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["method"] in METHODS:
                final[(int(r["budget_MB"]), r["method"])] = float(r["final_acc_mean"])

    fig, ax = plt.subplots(figsize=(7, 3.8))
    for name in METHODS:
        ys = [final[(b, name)] for b in BUDGETS]
        if all(y >= YMIN for y in ys):
            ax.plot(BUDGETS, ys, color=COLORS[name], marker=MARKERS.get(name, "o"),
                    lw=2 if name.startswith("Hybrid") else 1.5, label=name)
            continue
        # below the axis: arrows on the bottom edge with the real values
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
    ax.legend(ncol=3, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, 1.2), frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = os.path.join(args.fig_dir, f"fig_ptbxl_budget_summary_zoom.{ext}")
        fig.savefig(out, dpi=200)
        print("wrote", out)
    plt.close(fig)


if __name__ == "__main__":
    main()
