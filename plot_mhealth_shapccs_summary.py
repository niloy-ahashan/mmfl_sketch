"""Figures 4 and 6 of the paper (mHealth bars and final accuracy vs budget) with ShapCCS (B5) added.

Recreates the style of the original fig_mhealth_budget_bars.pdf / fig_mhealth_budget_summary.pdf (made
outside this repo) from the Table 5 (tab:mhealth_summary) values. ShapCCS = seed means from
ShapCCS/torch_port/result_updated/*mhealth*_summary.txt. FedMobile cannot finish a round within 5 MB.
Writes figures/fig_mhealth_budget_{bars,summary}_shapccs.pdf; the originals are untouched.
Run: venv/bin/python plot_mhealth_shapccs_summary.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Multimodal_Federated_Learning", "figures")
BUDGETS = [5, 10, 15, 20, 25]
NAN = float("nan")
VALS = {
    "B1 (IC)": [28.69, 46.25, 74.44, 77.94, 77.06],
    "B2 (MFedMC)": [18.08, 18.04, 18.06, 19.98, 19.35],
    "B3 (PFMH)": [24.83, 25.87, 29.58, 35.00, 38.94],
    "B4 (FedMobile)": [NAN, 19.50, 55.15, 69.40, 83.71],
    "B5 (ShapCCS)": [66.73, 81.74, 85.17, 88.96, 93.44],
    "HybridSketchMFedMC": [75.25, 93.63, 95.87, 96.85, 97.15],
}
# colour, hatch (bars), marker and line style (summary); B1-B4 and Hybrid as in the original figures
STYLE = {
    "B1 (IC)": ("#1f77b4", "///", "s", "-"),
    "B2 (MFedMC)": ("#ff7f0e", "---", "^", "--"),
    "B3 (PFMH)": ("#9467bd", "..", "D", ":"),
    "B4 (FedMobile)": ("#8c564b", "\\\\\\", "v", "-."),
    "B5 (ShapCCS)": ("#7f7f7f", "++", "X", (0, (5, 1.5, 1, 1.5, 1, 1.5))),
    "HybridSketchMFedMC": ("#2ca02c", "xx", "o", "-"),
}
RC = {"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9, "hatch.linewidth": 0.6,
      "pdf.fonttype": 42}


def style_axes(ax):
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.yaxis.grid(True, color="#dddddd", lw=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Cumulative upload budget (MB)", fontsize=10)
    ax.set_ylabel("Test accuracy (%)", fontsize=10)


def bars():
    fig, ax = plt.subplots(figsize=(3.5, 2.95))
    names = list(VALS)
    n, w = len(names), 0.84 / len(names)
    x = np.arange(len(BUDGETS))
    for i, name in enumerate(names):
        c, h, _, _ = STYLE[name]
        xs = x + (i - (n - 1) / 2) * w
        ax.bar(xs, np.nan_to_num(VALS[name]), w, color=c, edgecolor="black", hatch=h, lw=0.5, label=name, zorder=3)
        for xi, v in zip(xs, VALS[name]):
            if np.isnan(v):
                ax.text(xi, 2, "0 rounds", rotation=90, ha="center", va="bottom", fontsize=6, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(BUDGETS)
    style_axes(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7.5, frameon=False,
              handlelength=1.8, columnspacing=1.0, handletextpad=0.5, borderaxespad=0.2)
    fig.tight_layout(pad=0.3)
    out = os.path.join(FIG, "fig_mhealth_budget_bars_shapccs.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def summary():
    fig, ax = plt.subplots(figsize=(3.5, 2.95))
    order = ["HybridSketchMFedMC", "B1 (IC)", "B2 (MFedMC)", "B3 (PFMH)", "B4 (FedMobile)", "B5 (ShapCCS)"]
    for name in order:
        c, _, m, ls = STYLE[name]
        ax.plot(BUDGETS, VALS[name], color=c, marker=m, ls=ls, lw=1.6, ms=4.5, label=name,
                zorder=4 if name.startswith("Hybrid") else 3)
    ax.set_xticks(BUDGETS)
    style_axes(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7.5, frameon=False,
              handlelength=2.6, columnspacing=1.0, handletextpad=0.5, borderaxespad=0.2)
    fig.tight_layout(pad=0.3)
    out = os.path.join(FIG, "fig_mhealth_budget_summary_shapccs.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    with plt.rc_context(RC):
        print(bars())
        print(summary())
