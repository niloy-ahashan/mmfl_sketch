"""Seed summary (Results-sheet row) for the PTB-XL B2 logs written by main.py in this folder.

Runs MFedMC/UCI_HAR_v2/summarize_seeds.py unchanged, pointed at this folder's result_updated/ and at
the MFedMC_v2_<dataset>_mm_ log prefix, with the 2-modality count ("#Modalities" reads 1/2 (50%)).
Same flags; called at the end of run_ptbxl_mm.sh, e.g.
  python summarize_seeds.py --dataset ptbxl_mm_p44 --seeds 42 43 44 45 46
"""
import argparse
import importlib.util
import os
import sys

HERE = os.path.abspath(os.path.dirname(__file__))
UCI_DIR = os.path.abspath(os.path.join(HERE, "..", "UCI_HAR_v2"))

_spec = importlib.util.spec_from_file_location("uci_har_v2_summarize",
                                               os.path.join(UCI_DIR, "summarize_seeds.py"))
summ = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(summ)

pre = argparse.ArgumentParser(add_help=False)
pre.add_argument("--dataset", default="ptbxl_mm_p44")
dataset = pre.parse_known_args()[0].dataset
if "--dataset" not in sys.argv:
    sys.argv += ["--dataset", dataset]

summ.HERE = HERE  # default --log_dir = this folder's result_updated/
summ.PREFIX = f"MFedMC_v2_{dataset}"
summ.TOTAL_MODALITIES.update({"ptbxl_mm": 2, "ptbxl_mm_p44": 2})

if __name__ == "__main__":
    summ.main()
