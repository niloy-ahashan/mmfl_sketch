"""Plot the mean test accuracy over seeds of one ShapCCS config against cumulative upload, read from the
seed-tagged text logs in result_updated/ (same style as FedMobile/torch_port/plot_seed_mean_mhealth.py, so
the budget plots line up).

One curve = mean over the given seeds of test_acc at each round, with the individual seeds drawn faintly
underneath; y is fixed to 0-100 % in steps of 20. The std is not drawn; it is printed and written to the
.csv next to the PNG. After the selection round the number of clients per round depends on how many clients
each seed kept, so the cumulative upload can differ between seeds: x is the mean over seeds, and the spread
is printed. The vertical dotted line marks the selection round.

Pick the config with --dataset / --prune_frac / --rounds, e.g.
  venv/bin/python ShapCCS/torch_port/plot_seed_mean.py --dataset mhealth_mm --prune_frac 0.7 --rounds 100 \
      --seeds 42 43 44 45 46
"""
import argparse
import ast
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COLOR = "#8c5a2b"  # not used by any method in plot_mhealth_budget_comparison.py's COLORS
NAMES = {"mhealth": "mHealth", "uci_har": "UCI HAR"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", nargs="+", required=True)
    p.add_argument("--dataset", default="mhealth_mm", help="dataset dir name in the log names")
    p.add_argument("--prune_frac", type=float, required=True)
    p.add_argument("--client_ratio", type=float, default=0.2)
    p.add_argument("--rounds", type=int, default=None, help="only runs with this many rounds")
    p.add_argument("--log_dir", default=os.path.join(HERE, "result_updated"))
    p.add_argument("--out", default=None, help="PNG path (default: ShapCCS/torch_port/results/<config>_mean.png)")
    return p.parse_args()


def find_log(args, seed):
    r = args.rounds if args.rounds else "*"
    pat = os.path.join(args.log_dir, f"ShapCCS_{args.dataset}_mm_*MB_dirichlet*_it{r}_C{args.client_ratio:g}"
                                     f"_pf{args.prune_frac:g}_rem*_seed{seed}.txt")
    hits = [f for f in glob.glob(pat) if "NAMB" not in f and f.endswith(f"_seed{seed}.txt")]
    if len(hits) > 1:
        raise SystemExit(f"Several logs match seed {seed}; pass --rounds to pick one:\n  " + "\n  ".join(sorted(hits)))
    return hits[0] if hits else None


def table(path):
    """All rows of the per-round table, as {column: np.array}."""
    keys, rows = None, []
    for ln in open(path, encoding="utf-8"):
        parts = ln.split()
        if parts and parts[0] == "iter":
            keys, rows = parts, []
        elif keys and len(parts) == len(keys) and parts[0].isdigit():
            rows.append(list(map(float, parts)))
    if not rows:
        return None
    arr = np.array(rows)
    return {k: arr[:, i] for i, k in enumerate(keys)}


def header(path):
    """Resolved hyperparameters (as strings), the modality count and the kept clients."""
    hp, n_mod, kept = {}, None, None
    for ln in open(path, encoding="utf-8"):
        m = re.search(r"mod_dims=(\{.*?\})", ln) or re.search(r"Acc dim=\d+, Gyro dim=\d+", ln)
        if m:
            n_mod = len(ast.literal_eval(m.group(1))) if m.groups() else 2
        m = re.match(r"^  (\w+): (.+)$", ln)
        if m and hp.get("_done") is None:
            hp[m.group(1)] = m.group(2).strip()
        if ln.split()[:1] == ["iter"]:
            hp["_done"] = True
        m = re.match(r"^Kept clients after round \d+: (\[.*?\])", ln)
        if m:
            kept = ast.literal_eval(m.group(1))
    return hp, n_mod, kept


def main():
    args = parse_args()
    runs = {}
    for s in args.seeds:
        f = find_log(args, s)
        t = table(f) if f else None
        if t is None:
            raise SystemExit(f"No ShapCCS {args.dataset} pf{args.prune_frac:g} log with a round table for seed {s}")
        runs[s] = (f, t)

    lengths = {len(t["iter"]) for _, t in runs.values()}
    if len(lengths) > 1:
        raise SystemExit(f"Seeds have different round counts: { {s: len(t['iter']) for s, (_, t) in runs.items()} }")
    first_file, first = next(iter(runs.values()))
    ups = np.stack([t["cum_up_MB"] for _, t in runs.values()])
    x = ups.mean(axis=0)
    acc = np.stack([100 * t["test_acc"] for _, t in runs.values()])
    mean = acc.mean(axis=0)
    std = acc.std(axis=0, ddof=1) if len(runs) > 1 else np.zeros_like(mean)

    base = os.path.basename(first_file)
    stem = base[: base.rindex("_seed")]
    out = args.out or os.path.join(HERE, "results", f"{stem}_seeds{'_'.join(args.seeds)}_mean.png")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    hp, n_mod, _ = header(first_file)
    n_clients = int(hp["num_clients"])
    pre = int(hp["pre_train_round"])
    rem = int(hp["num_clients_remain"])
    kept_counts = [len(header(f)[2] or []) for f, _ in runs.values()]
    k0 = int(np.ceil(n_clients * float(hp["client_ratio"])))
    k_after = [int(t["clients"][-1]) for _, t in runs.values()]
    kind = "mhealth" if "mhealth" in args.dataset else "uci_har"

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    for xs, row in zip(ups, acc):
        ax.plot(xs, row, color=COLOR, alpha=0.18, linewidth=0.9)
    ax.plot(x, mean, color=COLOR, linewidth=2.0, label=f"mean of {len(runs)} seeds")
    if pre < len(x):
        ax.axvline(x[pre - 1], color="0.5", linestyle=":", linewidth=1)
    ax.annotate(f"{mean[-1]:.2f}%", xy=(x[-1], mean[-1]), xytext=(-8, 10), textcoords="offset points",
                ha="right", fontsize=9)
    ax.set_title(f"ShapCCS on {NAMES[kind]}, prune_frac {args.prune_frac:g}, {x[-1]:.2f} MB upload, "
                 f"{len(x)} rounds", fontsize=10)
    ax.text(0.98, 0.04,
            f"Clients: {k0}/{n_clients} per round, then {min(k_after)}"
            f"{'' if min(k_after) == max(k_after) else '-' + str(max(k_after))} of the kept\n"
            f"Kept after round {pre}: {min(kept_counts)}"
            f"{'' if min(kept_counts) == max(kept_counts) else '-' + str(max(kept_counts))} of {n_clients} "
            f"(tau = B/{rem})\n"
            f"Modalities: {n_mod}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.7"))
    ax.set_xlabel("Cumulative upload (MB)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_xlim(0, x[-1] * 1.02)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out)

    csv = os.path.splitext(out)[0] + ".csv"
    np.savetxt(csv, np.column_stack([first["iter"], x, mean, std]), delimiter=",",
               header="round,cum_up_MB,mean_test_acc_pct,std_test_acc_pct", comments="", fmt="%g")

    spread = ups[:, -1].max() - ups[:, -1].min()
    print(f"Seeds {' '.join(args.seeds)}: final round {int(first['iter'][-1])}, cum_up {x[-1]:.2f} MB "
          f"(spread over seeds {spread:.2f} MB), test acc {mean[-1]:.2f} ± {std[-1]:.2f} %")
    print(f"Plot saved to {out}")
    print(f"Data saved to {csv}")


if __name__ == "__main__":
    main()
