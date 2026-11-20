"""plot_mhealth_budget_comparison.py with ShapCCS (B5) added as a sixth method.

Imports the original script unchanged and only extends its method list, colours and markers, so the
plots keep the same look. Reads ShapCCS's seed-mean csv written by
  ShapCCS/torch_port/plot_seed_mean.py --dataset mhealth_mm --prune_frac 0.3 --rounds R --seeds ...
(R = 8/21/36/51/66 for 5/10/15/20/25 MB; the 5 MB run uses seeds 42 43 41 40 46, the others 42-46).

Figure 2 of the paper (FedMobile in place of FL-FD, plus ShapCCS), all five budgets:
  venv/bin/python plot_mhealth_budget_comparison_shapccs.py --all
-> results/mhealth_budget_comparison/mhealth_<b>MB_comparison_fedmobile_shapccs.{png,csv}
One budget: the original flags, e.g. --budget 5 --hybrid_cols 5000 --hybrid_client 0.2 --hybrid_top 3
--hybrid_rounds 42 --exclude FL-FD --suffix fedmobile_shapccs
"""
import sys

import plot_mhealth_budget_comparison as base

SHAPCCS_ROUNDS = {5: 8, 10: 21, 15: 36, 20: 51, 25: 66}
SHAPCCS_SEEDS = {5: "42 43 41 40 46"}
# Hybrid config per budget used in the paper's mHealth comparison (CLAUDE.md table).
HYBRID = {5: (5000, 0.2, 3, 42), 10: (5000, 0.2, 3, 84), 15: (15000, 0.1, 3, 84),
          20: (20000, 0.1, 3, 84), 25: (25000, 0.1, 3, 84)}

base.COLORS["ShapCCS"] = "#1a1a1a"   # black, as in the UCI HAR figures
base.MARKERS["ShapCCS"] = "X"
_methods = base.methods


def methods(args):
    b = args.budget
    r = SHAPCCS_ROUNDS[b]
    seeds = SHAPCCS_SEEDS.get(b, "42 43 44 45 46")
    return _methods(args) + [(
        "ShapCCS",
        f"ShapCCS/torch_port/results/ShapCCS_mhealth_mm_mm_*MB_dirichlet*_it{r}_C0.2_pf0.3_*"
        f"_seeds{seeds.replace(' ', '_')}_mean.csv",
        f"2/10 clients · 30% of data pruned · {r} rds",
        f"ShapCCS/torch_port/plot_seed_mean.py --dataset mhealth_mm --prune_frac 0.3 --rounds {r} --seeds {seeds}")]


base.methods = methods

if __name__ == "__main__":
    if sys.argv[1:] == ["--all"]:
        argv0 = sys.argv[0]
        for b, (cols, client, top, rounds) in HYBRID.items():
            sys.argv = [argv0, "--budget", str(b), "--hybrid_cols", str(cols), "--hybrid_client", str(client),
                        "--hybrid_top", str(top), "--hybrid_rounds", str(rounds),
                        "--exclude", "FL-FD", "--suffix", "fedmobile_shapccs"]
            base.main()
    else:
        base.main()
