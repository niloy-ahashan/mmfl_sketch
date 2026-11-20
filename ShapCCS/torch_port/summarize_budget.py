"""ShapCCS at a fixed cumulative upload budget, in the columns of the Results sheet.

How many clients train per round after selection depends on how many clients the seed keeps
(ceil(0.2 * kept): 1 for <= 5 kept, 2 for 6-10), so a fixed --rounds gives a different upload per seed.
This script takes one long run per seed (e.g. --rounds 200) and, for each seed, reads the round whose
cumulative upload is nearest to --budget_mb. Cutting a long run is exact: the learning rate depends only on
the round index (0.05 * 0.99^round) and selection happens at round 10, so rounds 0..r of a 200-round run are
the same as an (r+1)-round run with the same seed.

  MFedMC/.venv/bin/python ShapCCS/torch_port/summarize_budget.py --dataset uci_har_mm --prune_frac 0.3 \
      --rounds 200 --budget_mb 50 --seeds 42 43 44 45 46
"""
import argparse
import glob
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", nargs="+", required=True)
    p.add_argument("--dataset", default="uci_har_mm", help="dataset dir name in the file names")
    p.add_argument("--prune_frac", type=float, required=True)
    p.add_argument("--budget_mb", type=float, required=True, help="cumulative upload budget (MB = 1e6 bytes)")
    p.add_argument("--rounds", type=int, required=True, help="length of the long runs to cut (the R<n> in the npz name)")
    p.add_argument("--client_ratio", type=float, default=0.2)
    p.add_argument("--num_clients", type=int, default=10)
    p.add_argument("--results_dir", default=os.path.join(HERE, "results"))
    p.add_argument("--log_dir", default=os.path.join(HERE, "result_updated"))
    return p.parse_args()


def find_npz(args, seed):
    rx = re.compile(rf"^ShapCCS_{re.escape(args.dataset)}_K\d+_C{args.client_ratio:g}_pf{args.prune_frac:g}"
                    rf"_rem\d+_R{args.rounds}_seed{re.escape(seed)}\.npz$")
    hits = [f for f in glob.glob(os.path.join(args.results_dir, f"ShapCCS_{args.dataset}_*_seed{seed}.npz"))
            if rx.match(os.path.basename(f))]
    return max(hits, key=os.path.getmtime) if hits else None


def main():
    args = parse_args()
    runs, missing = [], []
    for s in args.seeds:
        f = find_npz(args, s)
        if f is None:
            missing.append(s)
            continue
        d = np.load(f)
        cum_up = np.cumsum(d["upload_bytes"]) / 1e6
        i = int(np.argmin(np.abs(cum_up - args.budget_mb)))  # round nearest the budget
        if i == len(cum_up) - 1 and cum_up[-1] < args.budget_mb:
            print(f"warning: seed {s} reaches only {cum_up[-1]:.2f} MB in {len(cum_up)} rounds; use a longer run")
        sel = d["selected"]
        runs.append(dict(seed=s, file=os.path.basename(f), r=i + 1, rem=int(d["num_clients_remain"]),
                         k0=int((sel[0] >= 0).sum()), k=int((sel[i] >= 0).sum()),
                         loss=float(d["test_loss"][i]), acc=100 * float(d["test_acc"][i]),
                         time=float(d["time_s"][i]), up=float(cum_up[i]), kept=d["kept_clients"].tolist()))
    if not runs:
        raise SystemExit(f"No R{args.rounds} ShapCCS results for {args.dataset}, prune_frac {args.prune_frac:g}")
    rem = runs[0]["rem"]

    def mean(key):
        return float(np.mean([r[key] for r in runs]))

    def std(key):
        v = np.array([r[key] for r in runs], dtype=float)
        return float(v.std(ddof=1)) if len(v) > 1 else 0.0

    n, k0 = args.num_clients, runs[0]["k0"]
    rounds_set = sorted({r["r"] for r in runs})
    rounds_str = str(rounds_set[0]) if len(rounds_set) == 1 else f"{rounds_set[0]}-{rounds_set[-1]}"
    name = f"ShapCCS (pf{args.prune_frac:g})"
    head = ["Methods", "#Rows", "#Cols", "#Clients", "#Modalities", "Total Rounds",
            "Avg. Test Loss", "Avg. Test Accuracy (%)", "Avg. Time (s)", "Total Upload"]
    row = [name, "-", "-", f"{k0}/{n} ({round(100 * k0 / n)}%)", "-", rounds_str,
           f"{mean('loss'):.4f}", f"{mean('acc'):.2f}", f"{mean('time'):.2f}", f"{mean('up'):.2f} MB"]

    used = [r["seed"] for r in runs]
    lines = [f"{name}, {args.dataset}, tau = B/{rem}, budget {args.budget_mb:g} MB "
             f"(cut from R{args.rounds} runs), mean over {len(runs)} seed(s): {' '.join(used)}",
             "", "Per seed (round nearest the budget):",
             f"{'seed':>6} {'round':>6} {'test loss':>10} {'test acc %':>11} {'time s':>9} {'upload MB':>10} "
             f"{'clients/rnd':>12}  kept clients"]
    for r in runs:
        lines.append(f"{r['seed']:>6} {r['r']:>6} {r['loss']:>10.4f} {r['acc']:>11.2f} {r['time']:>9.2f} "
                     f"{r['up']:>10.2f} {r['k0']:>5} -> {r['k']:<4}  {r['kept']}")
    lines += ["", "Results-sheet row (mean over seeds; #Clients = clients per round before selection):"]
    widths = [max(len(h), len(v)) for h, v in zip(head, row)]
    lines.append("  ".join(h.ljust(w) for h, w in zip(head, widths)))
    lines.append("  ".join(v.ljust(w) for v, w in zip(row, widths)))
    lines += ["", f"std over seeds: test loss {std('loss'):.4f}, test acc {std('acc'):.2f}, time {std('time'):.2f}, "
                  f"upload {std('up'):.2f}",
              "", "Tab-separated, paste into the sheet (columns A-J):", "\t".join(row)]
    if missing:
        lines += ["", f"No R{args.rounds} results file for seed(s): {' '.join(missing)} (not in the mean)"]
    text = "\n".join(lines)
    print("\n" + text)

    out = os.path.join(args.log_dir, f"ShapCCS_{args.dataset}_mm_budget{args.budget_mb:g}MB_from_R{args.rounds}"
                                     f"_C{args.client_ratio:g}_pf{args.prune_frac:g}_rem{rem}"
                                     f"_seeds{'_'.join(used)}_summary.txt")
    os.makedirs(args.log_dir, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\nSummary saved to {out}")


if __name__ == "__main__":
    main()
