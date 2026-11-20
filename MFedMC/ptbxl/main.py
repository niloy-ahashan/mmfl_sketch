"""MFedMC v2 (B2) on PTB-XL limb/chest features.

datasets/ptbxl_mm(_p44) has the same format as uci_har_mm (img_* / txt_* keys plus a cached
client*.npz / test.npz partition), so the algorithm is MFedMC/UCI_HAR_v2's own code, imported and run
as is (``_run`` with --partition dirichlet). ``load_from_sketchfusion_dir`` reuses the cached client
split, so the clients are the same as in B1 (IC) and B3 (Hybrid) on that cache. No UCI_HAR_v2 file is
modified, and nothing is written there:

Logs: MFedMC/ptbxl/result_updated/MFedMC_v2_<dataset>_mm_<up>MB_dirichlet0.1_it<N>_..._seed<s>.txt
npz:  MFedMC/ptbxl/results/<dataset>/MFedMC_v2_<dataset>_mm_<config>_seed<s>.npz
      (UCI_HAR_v2's _run saves it as MFedMC_UCI_HAR_v2_mm_<config>.npz in --results_dir; this
      script renames it right after)

Modalities: "Acc" = Limb leads (I, II, III, aVR, aVL, aVF), "Gyro" = Chest leads (V1-V6), 113-D each.
The console lines "Loading UCI HAR Acc/Gyro features..." and "official UCI HAR test set" are inherited
text; here the test set is PTB-XL strat folds 9-10.

Run via run_ptbxl_mfedmc_v2.sh (repo root) -> MFedMC/ptbxl/run_ptbxl_mm.sh.
"""

import glob
import importlib.util
import io
import os
import sys
import time

HERE = os.path.abspath(os.path.dirname(__file__))
UCI_DIR = os.path.abspath(os.path.join(HERE, "..", "UCI_HAR_v2"))
sys.path.insert(0, UCI_DIR)  # UCI_HAR_v2's main.py imports options/dataset/federated/models from there

_spec = importlib.util.spec_from_file_location("uci_har_v2_main", os.path.join(UCI_DIR, "main.py"))
uci = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(uci)

UCI_PREFIX = "MFedMC_UCI_HAR_v2_mm_"


def _dataset_name(args):
    return os.path.basename(os.path.normpath(args.dataset_dir or "ptbxl_mm"))


def _txt_stem(args, cum_up_mb):
    stem = uci._txt_stem(args, cum_up_mb)
    assert stem.startswith(UCI_PREFIX), stem
    return f"MFedMC_v2_{_dataset_name(args)}_mm_" + stem[len(UCI_PREFIX):]


def _rename_npz(args, since):
    """Give the npz _run just wrote a PTB-XL name with the seed in it."""
    for f in glob.glob(os.path.join(args.results_dir, UCI_PREFIX + "*.npz")):
        if os.path.getmtime(f) >= since:
            base = os.path.basename(f)[len(UCI_PREFIX):-len(".npz")]
            dst = os.path.join(args.results_dir,
                               f"MFedMC_v2_{_dataset_name(args)}_mm_{base}_seed{args.seed}.npz")
            os.replace(f, dst)
            print(f"Results renamed to {dst}")


def main():
    args = uci.args_parser()
    if not os.path.isabs(args.results_dir):
        args.results_dir = os.path.join(HERE, args.results_dir, _dataset_name(args))
    log_buf = io.StringIO()
    stdout_orig = sys.stdout
    sys.stdout = uci._Tee(stdout_orig, log_buf)
    cum_up_mb = None
    start = time.time()
    try:
        cum_up_mb = uci._run(args)
        _rename_npz(args, start)
    finally:
        sys.stdout = stdout_orig
        updated_dir = os.path.join(HERE, "result_updated")
        os.makedirs(updated_dir, exist_ok=True)
        txt_path = os.path.join(updated_dir, _txt_stem(args, cum_up_mb) + ".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(log_buf.getvalue())
        print(f"Console log saved to {txt_path}")


if __name__ == "__main__":
    main()
