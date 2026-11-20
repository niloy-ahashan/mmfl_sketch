#!/usr/bin/env python3
"""
Centralized sanity check of the PTB-XL feature cache (``prepare_ptbxl_mm.py``).

No federation: fits simple classifiers on all training data, to see how much
diagnostic signal the hand-crafted features carry before any baseline runs on
them. Compare against the majority-class row, since NORM dominates.

Usage
-----
  python CommEfficient/CommEfficient/check_ptbxl_mm.py --data_dir datasets/ptbxl_mm
"""

from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.neural_network import MLPClassifier


def _models(seed: int) -> dict:
    return {
        "majority": DummyClassifier(strategy="most_frequent"),
        "logreg": LogisticRegression(max_iter=2000),
        "boosting": HistGradientBoostingClassifier(random_state=seed),
        "mlp_512_512": MLPClassifier((512, 512), early_stopping=True, max_iter=200,
                                     random_state=seed),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="PTB-XL feature cache classifier check")
    p.add_argument("--data_dir", type=str, default="datasets/ptbxl_mm")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    raw = np.load(os.path.join(args.data_dir, "data.npz"))
    with open(os.path.join(args.data_dir, "prepare_stats.json"), encoding="utf-8") as f:
        names = json.load(f)["class_names"]
    y_tr, y_te = raw["labels_train"], raw["labels_test"]
    inputs = {
        "limb": (raw["img_train"], raw["img_test"]),
        "chest": (raw["txt_train"], raw["txt_test"]),
        "both": (np.hstack([raw["img_train"], raw["txt_train"]]),
                 np.hstack([raw["img_test"], raw["txt_test"]])),
    }

    recall_hdr = "".join(f"{n:>7}" for n in names)
    print(f"{'input':<7}{'model':<13}{'acc':>7}{'macroF1':>9}  recall:{recall_hdr}{'time':>8}")
    results = []
    best = None
    for input_name, (x_tr, x_te) in inputs.items():
        for model_name, model in _models(args.seed).items():
            start = time.time()
            pred = model.fit(x_tr, y_tr).predict(x_te)
            acc = accuracy_score(y_te, pred)
            macro_f1 = f1_score(y_te, pred, average="macro", zero_division=0)
            recall = recall_score(y_te, pred, average=None, labels=range(len(names)),
                                  zero_division=0)
            secs = time.time() - start
            recall_txt = "".join(f"{r:>7.3f}" for r in recall)
            print(f"{input_name:<7}{model_name:<13}{acc:>7.3f}{macro_f1:>9.3f}"
                  f"         {recall_txt}{secs:>7.0f}s", flush=True)
            row = {"input": input_name, "model": model_name, "accuracy": float(acc),
                   "macro_f1": float(macro_f1),
                   "recall": dict(zip(names, recall.tolist())), "seconds": secs}
            results.append(row)
            if best is None or macro_f1 > best[0]:
                best = (macro_f1, row, confusion_matrix(y_te, pred, labels=range(len(names))))

    _, row, cm = best
    print(f"\nHighest macro-F1: {row['input']} / {row['model']}; "
          "confusion matrix (rows = true, cols = predicted):")
    print(" " * 6 + "".join(f"{n:>6}" for n in names))
    for name, line in zip(names, cm):
        print(f"{name:<6}" + "".join(f"{v:>6}" for v in line))

    out = os.path.join(args.data_dir, "check_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "results": results,
                   "best": {"input": row["input"], "model": row["model"],
                            "confusion_matrix": cm.tolist()}}, f, indent=1)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
