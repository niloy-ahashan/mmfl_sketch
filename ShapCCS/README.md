# ShapCCS baseline

Ji, Hu, He, Zhao, Li, Xu. *ShapCCS: Shapley-Driven Client Coreset Selection in Federated Learning.* ICML 2026.
OpenReview: https://openreview.net/forum?id=PnabunIVDL (paper: `ShapCCS.pdf`, fetched through Chrome because
OpenReview blocks command-line downloads).

- `upstream/`: https://github.com/iz70778/ShapCCS at commit `8a45bbbed4ebe78e4913e0cb44bfe93460ffc6e1`
  (2026-03-31), unchanged (`.git` removed). `A1.pdf`–`C2.pdf` are the rebuttal figures.
- `torch_port/`: the released `fed_sv_cs_super` + Moderate path on the shared UCI HAR / mHealth split.
  - `main.py`: entry point; docstring lists every setting that differs from upstream and why.
  - `gpsv.py`: upstream's selection and training functions, copied; changed lines are marked `# PORT:`.
  - `dataset_mm.py`: Hybrid v2's loaders wrapped in upstream's `PrunableDataset` interface.
  - `check_port.py`: compares `gpsv.py` against the upstream modules on identical inputs (no dataset needed).
  - `run_uci_har.sh`, `run_mhealth.sh`: 5 seeds × prune_frac 0.7 (τ = B/2) and 0.3 (τ = B/7), client ratio 0.2.
  - `summarize_seeds.py`, `plot_seed_mean.py`: Results-sheet row and seed-mean curve + `*_mean.csv`.

```bash
MFedMC/.venv/bin/python ShapCCS/torch_port/check_port.py   # port == upstream
bash run_mhealth_shapccs.sh                                 # or run_uci_har_shapccs.sh
```
