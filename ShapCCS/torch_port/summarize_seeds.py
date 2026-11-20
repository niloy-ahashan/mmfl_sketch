"""Average ShapCCS runs over seeds, in the columns of the Results sheet.

One row = mean over the given seeds of the final round of each run:
  Avg. Test Loss / Accuracy = last-round shard test_loss / test_acc (the "Final test accuracy" line),
  Avg. Time (s)             = last-round time_s (training time, as in the log table),
  Total Upload              = cumulative upload MB.
Also listed per seed: clients kept at the selection round and the samples they keep.

Called automatically by run_uci_har.sh / run_mhealth.sh once per prune fraction; by hand, e.g.
  MFedMC/.venv/bin/python ShapCCS/torch_port/summarize_seeds.py --dataset mhealth_mm --prune_frac 0.7 \
      --seeds 42 43 44 45 46
  MFedMC/.venv/bin/python ShapCCS/torch_port/summarize_seeds.py --dataset uci_har_mm --prune_frac 0.3 \
      --rounds 100 --seeds 42
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
    p.add_argument("--dataset", default="mhealth_mm", help="dataset dir name in the file names")
    p.add_argument("--prune_frac", type=float, required=True, help="only runs with this prune fraction")
    p.add_argument("--client_ratio", type=float, default=0.2)
    p.add_argument("--rounds", type=int, default=None, help="only runs with this many rounds")
    p.add_argument("--extra", default="", help="extra name tag(s) after rem<k>, e.g. 'random' or 'fd32'; '' = none")
    p.add_argument("--since", type=float, default=0.0, help="only npz files written after this unix time")
    p.add_argument("--num_clients", type=int, default=10)
    p.add_argument("--results_dir", default=os.path.join(HERE, "results"))
    p.add_argument("--log_dir", default=os.path.join(HERE, "result_updated"))
    return p.parse_args()


def find_npz(args, seed):
    r = args.rounds if args.rounds else r"\d+"
    extra = f"_{re.escape(args.extra)}" if args.extra else ""
    rx = re.compile(rf"^ShapCCS_{re.escape(args.dataset)}_K\d+_C{args.client_ratio:g}_pf{args.prune_frac:g}"
                    rf"_rem\d+{extra}_R{r}_seed{re.escape(seed)}\.npz$")
    hits = [f for f in glob.glob(os.path.join(args.results_dir, f"ShapCCS_{args.dataset}_*_seed{seed}.npz"))
            if rx.match(os.path.basename(f)) and os.path.getmtime(f) >= args.since]
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
        per_round = d["selected"]
        k_final = int((per_round[-1] >= 0).sum())
        runs.append(dict(seed=s, file=os.path.basename(f), rounds=len(d["test_acc"]), rem=int(d["num_clients_remain"]),
                         k0=int((per_round[0] >= 0).sum()), k=k_final, loss=float(d["test_loss"][-1]),
                         acc=100 * float(d["test_acc"][-1]), time=float(d["time_s"][-1]),
                         up=float(d["total_upload_mib"]), kept=d["kept_clients"].tolist(),
                         kept_samples=int(d["kept_samples"].sum()), total=int(d["samples_per_client"].sum())))
    if not runs:
        raise SystemExit(f"No ShapCCS results found for {args.dataset}, prune_frac {args.prune_frac:g}, "
                         f"seeds {' '.join(args.seeds)}")
    configs = {(r["rounds"], r["rem"]) for r in runs}
    if len(configs) > 1:
        raise SystemExit(f"Seeds come from different configs (rounds, num_clients_remain): {sorted(configs)}; "
                         f"pass --rounds (or --since) to pick one")
    rounds, rem = configs.pop()

    def mean(key):
        return float(np.mean([r[key] for r in runs]))

    def std(key):
        v = np.array([r[key] for r in runs], dtype=float)
        return float(v.std(ddof=1)) if len(v) > 1 else 0.0

    n = args.num_clients
    k0 = runs[0]["k0"]
    name = f"ShapCCS (pf{args.prune_frac:g})"
    head = ["Methods", "#Rows", "#Cols", "#Clients", "#Modalities", "Total Rounds",
            "Avg. Test Loss", "Avg. Test Accuracy (%)", "Avg. Time (s)", "Total Upload"]
    row = [name, "-", "-", f"{k0}/{n} ({round(100 * k0 / n)}%)", "-", str(rounds),
           f"{mean('loss'):.4f}", f"{mean('acc'):.2f}", f"{mean('time'):.2f}", f"{mean('up'):.2f} MB"]

    used = [r["seed"] for r in runs]
    lines = [f"{name}, {args.dataset}, tau = B/{rem}, mean over {len(runs)} seed(s): {' '.join(used)}",
             "", "Per seed (final round):",
             f"{'seed':>6} {'test loss':>10} {'test acc %':>11} {'time s':>9} {'upload MB':>10} "
             f"{'clients/rnd':>12} {'samples kept':>13}  kept clients / file"]
    for r in runs:
        lines.append(f"{r['seed']:>6} {r['loss']:>10.4f} {r['acc']:>11.2f} {r['time']:>9.2f} {r['up']:>10.2f} "
                     f"{r['k0']:>5} -> {r['k']:<4} {r['kept_samples']:>6}/{r['total']:<6}  {r['kept']}  {r['file']}")
    lines += ["", "Results-sheet row (mean over seeds; #Clients = clients per round before selection):"]
    widths = [max(len(h), len(v)) for h, v in zip(head, row)]
    lines.append("  ".join(h.ljust(w) for h, w in zip(head, widths)))
    lines.append("  ".join(v.ljust(w) for v, w in zip(row, widths)))
    lines += ["", f"std over seeds: test loss {std('loss'):.4f}, test acc {std('acc'):.2f}, time {std('time'):.2f}, "
                  f"upload {std('up'):.2f}",
              "", "Tab-separated, paste into the sheet (columns A-J):", "\t".join(row)]
    if missing:
        lines += ["", f"No results file for seed(s): {' '.join(missing)} (not in the mean)"]
    text = "\n".join(lines)
    print("\n" + text)

    out = os.path.join(args.log_dir, f"ShapCCS_{args.dataset}_mm_{mean('up'):.2f}MB_it{rounds}_C{args.client_ratio:g}"
                                     f"_pf{args.prune_frac:g}_rem{rem}{'_' + args.extra if args.extra else ''}"
                                     f"_seeds{'_'.join(used)}_summary.txt")
    os.makedirs(args.log_dir, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\nSummary saved to {out}")


if __name__ == "__main__":
    main()
