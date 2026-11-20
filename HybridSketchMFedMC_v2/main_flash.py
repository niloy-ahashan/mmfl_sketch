"""HybridSketchMFedMC v2 on FLASH (GPS, Image, LiDAR) -- new entry point, Hybrid files untouched.

The whole algorithm is Hybrid v2's own code, imported and run as is: ``main._run``, ``federated.run_federated``
and ``options.args_parser`` (same flags).  In memory only, ``load_uci_har_mm`` inside Hybrid's main module is
replaced by the FLASH loader below, the path ``_run`` takes for any --dataset other than mhealth.

Data: datasets/flash_mm or datasets/flash_mm_episode, built by CommEfficient/CommEfficient/prepare_flash_mm.py.
The cached client*.npz / test.npz are required and never re-partitioned, so the client split stays the one IC
uses. Modality names, dims and num_classes come from the cache's prepare_stats.json.

Logs: HybridSketchMFedMC_v2/result_updated/HybridV2_<cache>_mm_<up>MB_..._seed<s>.txt
npz:  HybridSketchMFedMC_v2/results/flash/<same name>.npz  (its own folder: _run names its npz "UCI_HAR" on
      this path, which must not overwrite a real UCI HAR result in results/)
"""

import glob
import io
import json
import os
import sys
import time

import numpy as np

HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, HERE)

import main as hybrid  # noqa: E402  (Hybrid v2's main.py, unmodified)
from dataset import _cache_matches, _pack  # noqa: E402

DATASET = "flash"
RESULTS_SUBDIR = "flash"
# _run prints this on the path we borrow; it is not true here.
BORROWED_PATH_LINE = "Loading SketchFusionB UCI HAR split"


def _print_without_borrowed_line(*a, **kw):
    if not (a and str(a[0]).startswith(BORROWED_PATH_LINE)):
        print(*a, **kw)


def _flash_loader(args):
    def load(dataset_dir, num_clients=10, dirichlet_alpha=0.1, seed=42):
        dataset_dir = os.path.abspath(dataset_dir)
        stats_path = os.path.join(dataset_dir, "prepare_stats.json")
        if not (os.path.isfile(stats_path) and _cache_matches(dataset_dir, num_clients)):
            raise FileNotFoundError(
                f"Missing the {num_clients}-client FLASH cache in {dataset_dir}. Build it with "
                "CommEfficient/CommEfficient/prepare_flash_mm.py."
            )
        with open(stats_path, encoding="utf-8") as f:
            stats = json.load(f)
        modalities = list(stats["modalities"])
        n_mod = len(modalities)

        def read(name):
            d = np.load(os.path.join(dataset_dir, name))
            return _pack([d[f"m{k}_feats"] for k in range(n_mod)], d["labels"])

        clients = [read(f"client{i}.npz") for i in range(num_clients)]
        global_test = read("test.npz")
        n_per = [int(len(c["y"])) for c in clients]
        mod_dims = [int(x.shape[1]) for x in global_test["xs"]]
        # _run compares each modality to args.<name>_dim; these modalities have no flags of their own.
        for name, dim in zip(modalities, mod_dims):
            setattr(args, f"{name.lower()}_dim", dim)

        print(f"Loading FLASH split: reusing the client cache in {dataset_dir} ({num_clients} clients).")
        print(f"  mod_dims={dict(zip(modalities, mod_dims))}, classes={stats['num_classes']}, "
              f"train={sum(n_per)}, test={len(global_test['y'])}")
        print(f"  samples/client: {n_per} (min={min(n_per)}, max={max(n_per)})")
        meta = {
            "dataset_dir": dataset_dir,
            "reused_sketchfusion_cache": True,
            "num_clients": num_clients,
            "dirichlet_alpha": dirichlet_alpha,
            "mod_dims": mod_dims,
            "modalities": modalities,
            "samples_per_client": n_per,
            "num_classes": int(stats["num_classes"]),
            "num_test": int(len(global_test["y"])),
        }
        return clients, global_test, meta
    return load


def _seed_tag_npz(args, cum_up_mb, since):
    """_run's npz name has no seed: rename this run's file to the same name as its log."""
    hits = [f for f in glob.glob(os.path.join(args.results_dir, "HybridV2_UCI_HAR_*.npz"))
            if os.path.getmtime(f) >= since]
    if not hits:
        return
    dst = os.path.join(args.results_dir, hybrid._txt_stem(args, cum_up_mb) + ".npz")
    os.replace(max(hits, key=os.path.getmtime), dst)
    print(f"Results renamed to {dst}")


def main():
    args = hybrid.args_parser()  # Hybrid v2's own flags and defaults
    args.dataset = DATASET
    args.acc_dim = args.gyro_dim = None  # uci_har defaults from the parser; these are not modalities here
    if not os.path.isabs(args.results_dir):
        args.results_dir = os.path.join(HERE, args.results_dir, RESULTS_SUBDIR)
    hybrid.load_uci_har_mm = _flash_loader(args)
    hybrid.print = _print_without_borrowed_line

    log_buf = io.StringIO()
    stdout_orig = sys.stdout
    sys.stdout = hybrid._Tee(stdout_orig, log_buf)
    cum_up_mb = None
    start = time.time()
    try:
        cum_up_mb = hybrid._run(args)  # Hybrid v2's run, unchanged
        _seed_tag_npz(args, cum_up_mb, start)
    finally:
        sys.stdout = stdout_orig
        updated_dir = hybrid._result_updated_dir()
        os.makedirs(updated_dir, exist_ok=True)
        txt_path = os.path.join(updated_dir, hybrid._txt_stem(args, cum_up_mb) + ".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(log_buf.getvalue())
        print(f"Console log saved to {txt_path}")


if __name__ == "__main__":
    main()
