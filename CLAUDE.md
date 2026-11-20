# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Research code for communication-efficient **multimodal federated learning**. The proposed method is
**HybridSketchMFedMC_v2** (MFedMC client/modality selection + FetchSGD gradient sketches). It is
compared against baselines on a shared client partition and metric (see "Shared protocol" below):

| ID | Method | Folder | Root launcher (UCI HAR) | Log prefix |
|---|---|---|---|---|
| B1 | IndependentCompression (IC) + FetchSGD | `CommEfficient/` | `run_uci_har_independent_compression.sh` | `IC_` |
| B2 | MFedMC v2 | `MFedMC/UCI_HAR_v2/`, `MFedMC/mHealth_v2/` | `run_uci_har_mfedmc_sketchfusion_protocol_v2.sh` | `MFedMC_UCI_HAR_v2_`, `MFedMC_mHealth_v2_` |
| B3 | **HybridSketchMFedMC v2 (ours)** | `HybridSketchMFedMC_v2/` | `run_uci_har_hybrid_sketch_mfedmc_v2.sh` | `HybridV2_` |
| B4 | PMFH (+ variants) | `PMFH/` | `PMFH/run_uci_har_pmfh*.sh` | `PMFH_`, `PMFHv2_`, `PMFHfh_`, `PMFHjh_`, mHealth: `PMFHfhN`, `PMFHfhG` |
| — | FLASH (PyTorch port) | `FLASH/torch_port/` | `run_uci_har_flash.sh` | `FLASH_` |
| — | FL-FD | `FL-FD/fixed/` (UP-Fall), `FL-FD/uci_har/`, `FL-FD/mhealth/` | `run_upfall_flfd.sh`, `run_uci_har_flfd_features.sh` | `FLFD_UPFall_`, `FLFD_<dataset>_` |
| — | FedMobile (PyTorch port) | `FedMobile/torch_port/` | `run_uci_har_fedmobile.sh` | `FedMobile_` |
| — | ShapCCS (port of the released code) | `ShapCCS/torch_port/` | `run_uci_har_shapccs.sh` | `ShapCCS_` |
| — | PEPSY (upstream, PTB-XL only, not on the shared protocol) | `PEPSY/` | `PEPSY/run_ptbxl.sh` | `PEPSY_ptbxl_` |
| — | Hybrid v2 + FL-FD small CNN (experimental) | `HybridSketchMFedMC_v2_flfd/` | `run_uci_har_hybrid_sketch_mfedmc_v2_flfd.sh` | `HybridV2FD_` |

The "B1–B4" names match the tmux sessions / Cursor terminals the user runs them in. The paper draft
numbers baselines differently (its "B4" is FedMobile).

Supporting code:
- **`csh/`** — the standalone `csvec` Count Sketch library used by CommEfficient.
- **`HybridSketchMFedMC/`** (v1), **`MFedMC/UCI_HAR/`**, **`MFedMC/mHealth/`** (v1),
  **`MFedMC/uci_har/`** — earlier versions, kept because older run scripts/results reference them.
- **`MFedMC/ActionSense/`** — the original upstream MFedMC code (no runner; run `main.py` directly).
- **`MFedMC/UCI_HAR_paper/`** — paper reproduction (see B2).
- **`Multimodal_Federated_Learning/`** — the ACM paper draft (`main.tex`, `figures/`), untracked.

## Working conventions

These come from the user's own instructions in past sessions.

- **Do not modify existing scripts or existing Hybrid files.** Add a new script or file instead;
  existing ones are referenced by prior results. Delete only when asked.
- **Baseline ports keep the algorithm.** Change only data and logging adapters. Where released code
  and the paper differ, follow the released code and document the difference.
- **Adding a baseline:** vendor the paper and upstream code, show the workflow visually, verify the
  port, and only then add runners, a seed sweep and a summarizer.
- **Paper reproductions are a separate track.** They keep upstream behaviour and logging.
- **Runner values are user state.** The user hand-edits `--rounds`, `NUM_SEEDS` and similar between
  runs, so the numbers in a script may be mid-experiment rather than canonical. Preserve them.
- **Re-read files before explaining them.** Scripts change between turns, even within a session.
- **Parallel sessions share context.** Work is often split across sibling Claude sessions; check
  their transcripts before saying something is missing.
- **Long runs:** the user starts these in their own terminal, inside tmux.
- **Commit only when asked.**

## Setup

```bash
source venv/bin/activate                 # main interpreter for CommEfficient/csh
pip install -r requirements.txt
cd csh && pip install -e . && cd ..      # csvec, editable
cd CommEfficient && pip install -e . && cd ..
```

Interpreters differ per runner — check the `PYTHON=` logic in the script before assuming:
- `MFedMC/.venv/bin/python` if present, else `python3`: MFedMC v2, MFedMC paper, Hybrid v2, FLASH port,
  FedMobile port, ShapCCS port.
- root `venv/bin/python`: PMFH, all FL-FD folders, Hybrid v2 FL-FD (`MFedMC/.venv` lacks pyts and
  Pillow), PEPSY (hardcoded, with `PYTHONPATH=PEPSY/deps`), and plotting (`MFedMC/.venv` has no
  matplotlib).
- plain `python3` (with `PYTHONPATH=CommEfficient/`): IC and other CommEfficient scripts.

## Common commands

**Prepare a dataset** (raw data → feature cache consumed by training). Raw UCI HAR is at the repo root
in `human+activity+recognition+using+smartphones/`; other raw data is under `datasets/`.
```bash
python CommEfficient/CommEfficient/prepare_uci_har_mm.py --uci_root <raw_dir> \
  --out_dir datasets/uci_har_mm                                 # both flags required
python CommEfficient/CommEfficient/prepare_mhealth_mm.py --test_split random_group \
  --out_dir datasets/mhealth_mm_random_group                    # mHealth (4 modalities)
```
Each dataset has a `prepare_*.py` in `CommEfficient/CommEfficient/`; many are kept for reference with no
cache or launcher. FL-FD builds its own UP-Fall cache (`FL-FD/fixed/prepare_upfall.py`, called
automatically by its runner).

**Run an experiment** — always via a `run_*.sh` script, not by invoking the training script
directly; the scripts set `PYTHONPATH`/interpreter and pin the full flag set for that config:
```bash
bash run_uci_har_independent_compression.sh          # B1
bash run_uci_har_mfedmc_sketchfusion_protocol_v2.sh  # B2
bash run_uci_har_hybrid_sketch_mfedmc_v2.sh          # B3 (ours)
bash PMFH/run_uci_har_pmfh_fedhead.sh                # B4
bash run_uci_har_flash.sh                            # FLASH
bash run_uci_har_flfd_features.sh                    # FL-FD on UCI HAR (feature-vector version)
bash run_upfall_flfd.sh                              # FL-FD (UP-Fall)
bash run_uci_har_fedmobile.sh                        # FedMobile: currently the _miss runner, see below
bash run_uci_har_shapccs.sh                          # ShapCCS: prune_frac 0.7 and 0.3, 5 seeds each
```
mHealth counterparts: `run_mhealth_independent_compression_4mod_v2.sh`,
`run_mhealth_mfedmc_sketchfusion_protocol_v2.sh`, `run_mhealth_hybrid_sketch_mfedmc_v2.sh`,
`PMFH/run_mhealth_pmfh_fedhead_nmod.sh`, `run_mhealth_flash.sh`, `run_mhealth_flfd_features.sh`,
`run_mhealth_fedmobile.sh`, `run_mhealth_shapccs.sh`. Other root `run_*.sh` files are older configs (CV, MIR-Flickr, SketchFusion
A/B/C, v1 MFedMC/Hybrid).

FLASH dataset (Hybrid and IC only so far; `DATA_PATH=datasets/flash_mm_episode` selects the leakage check):
```bash
venv/bin/python CommEfficient/CommEfficient/prepare_flash_mm.py --out_dir datasets/flash_mm
venv/bin/python CommEfficient/CommEfficient/prepare_flash_mm.py --test_split episode --out_dir datasets/flash_mm_episode
bash run_flash_hybrid_sketch_mfedmc_v2.sh
bash run_flash_independent_compression.sh
```
IC's rendezvous port is hardcoded (`tcp://127.0.0.1:29500` in `fed_aggregator.py` and `fed_worker.py`), so only one
CommEfficient run can be alive at a time; a second fails with `EADDRINUSE`.

Extra args are forwarded to the Python entry point, but the round flag differs: `--num_epochs` (B1, B3),
`--iterations` (B2), `--rounds` (the others). `run_mhealth_independent_compression_4mod.sh` forwards
nothing. In a sweep, a forwarded `--seed` overrides every seed in the loop.

Root-level `run_*.sh` for Hybrid v2 / FLASH / FL-FD / FedMobile are one-line `exec` wrappers around the
real script inside the method's folder — edit flags there. Note `run_uci_har_fedmobile.sh` currently
execs `FedMobile/torch_port/run_uci_har_miss.sh`, while `run_mhealth_fedmobile.sh` execs the complete-data
`run_mhealth.sh`.

**Seed sweeps.** Runners take `NUM_SEEDS` and `SEEDS` as environment variables, usually seeds 42–46:

| Behaviour with no env vars | Runners |
|---|---|
| 5 seeds | B2 and B3 on mHealth, PMFH `fedhead` and `nmod`, FL-FD `uci_har` features and `mhealth`, all FedMobile, ShapCCS (for each of `PRUNE_FRACS="0.7 0.3"`) |
| 5 seeds, values `47 48 44 45 46` | `run_mhealth_independent_compression_4mod_v2.sh` |
| Seed 42; set the env vars to sweep (B1 defaults to `NUM_SEEDS=1`, B2 and B3 to 5) | B1, B2 and B3 on UCI HAR |
| 1 seed | `PMFH/run_mhealth_pmfh_fedhead_grouped.sh`, `HybridSketchMFedMC_v2_flfd/run_uci_har.sh` |
| No sweep logic | FLASH, `FL-FD/fixed/`, other PMFH UCI runners |

A failed seed must not stop a sweep, and each sweep should end with a summary. When writing a new
runner, pass the runner's actual dataset to the summarizer: a hardcoded `--dataset` that mismatches
`DATA_PATH`, combined with `|| true`, silently produces no summary table.

**Tests:**
```bash
python -m unittest csh/csvec/test_csvec.py    # real unittest suite for the sketch library
```
`CommEfficient/CommEfficient/{grad_test,unit_test,test_gradients,pytorch_test}.py` are standalone
diagnostic/debug scripts, not a CI-style suite.

## Datasets (`datasets/`)

| Folder | Modalities (dims) | Classes | Split |
|---|---|---|---|
| `uci_har_mm` | Acc 348, Gyro 213 (keys still `img_*`/`txt_*`) | 6 | official test set, 2947 samples |
| `uci_har_lstm_mm` | `body_acc`, `body_gyro` (128×3) | 6 | official test set |
| `mhealth_mm` | Acc 177, Gyro 118, Mag 118, ECG 43 | 12 | subjects 1–8 train, 9–10 test (4188/1035) |
| `mhealth_mm_random_group` | same | 12 | 70/30 split by group (3681/1542) |
| `mhealth_mm_random` | same | 12 | 70/30 random split (3656/1567) |
| `ptbxl_reduce` | 12 ECG leads (1000×12) | 5 | folds 1–8 train, 9–10 test (3170/800); PEPSY only |
| `flash_mm` | GPS 2, Image 512 (ResNet18), LiDAR 1006 | 49 | random 90/10 within each episode (28268/3244); near-duplicate frames can cross the split |
| `flash_mm_episode` | same | 49 | all episodes numbered 9 held out (28512/3000) |

`datasets/flash_processed/` holds the FLASH zip and `features_resnet18.npz` (features before splitting, reused by
`prepare_flash_mm.py`). The zip's `rf` has 64 columns: 411 all-zero rows are dropped and the 49 sectors that occur are
remapped, so the cache has 31,512 of the 31,923 samples and differs from the paper's 34 sectors. Stored frames are BGR.
The repository blocks command-line downloads (HTTP 403), so the zip was fetched through Chrome.

`datasets/mhealth+dataset/` and `datasets/ptb-xl-*/` hold raw data.

## Shared protocol (all UCI HAR / mHealth comparisons)

- **mHealth split:** `mhealth_mm` is the main cache, but runner defaults are mixed, so check
  `DATA_PATH` before comparing numbers:
  - `mhealth_mm`: IC `_4mod_v2`, B2, B3, PMFH `nmod`, FL-FD, FedMobile `run_mhealth_miss.sh`.
  - `mhealth_mm_random_group`: IC `_4mod`, PMFH `grouped`, FLASH, FedMobile `run_mhealth.sh`.
- **Clients:** 10-client Dirichlet(α=0.1) label-skew partition, all client data is training data
  (except FLASH mHealth, which holds out `--val_frac 0.1`); evaluation on the shared test set.
- **Metric:** `test_acc` = mean accuracy over 8-sample test shards (`valid_batch_size 8`), the IC
  convention. B2, B3, FLASH, FedMobile and PMFH-head variants import/replicate this so numbers are
  comparable. Not comparable: official PMFH (kNN on hash codes) and the MFedMC paper reproduction.
- **Communication:** bytes / 1e6 reported as "MB" (some print strings still say MiB — it's MB).
  Both upload and download are counted per round; the cumulative upload MB is embedded in the log
  filename. Exception: `FL-FD/fixed/` prints true MiB, tracks no download, and has no MB in its filename.
- **Logs:** each method tees its console table to `<method>/result_updated/<prefix>_..._<cum_up>MB_
  ..._seed<N>.txt` (IC writes to `CommEfficient/result_updated/`); `.npz` histories go to
  `results/` (IC writes no `.npz`; its `results/` holds seed-mean CSVs and plots). Both are gitignored.

## Reporting and plots

- **Results:** rows go in the user's "Results" Google Sheet with columns Methods, #Rows, #Cols, #Clients,
  #Modalities, Total Rounds, Avg Test Loss, Avg Test Acc, Avg Time, Total Upload. Give rows in chat to
  paste rather than writing to the sheet. The headline accuracy is final-round shard accuracy averaged
  over 5 seeds. Avg Time excludes initialization (e.g. GAF building).
- **Logs stay minimal:** extra metrics are commented out of the table but still computed for the npz.
- **Seed summaries:** `summarize_seeds.py` in `CommEfficient/CommEfficient/`, `MFedMC/UCI_HAR_v2/`,
  `MFedMC/mHealth_v2/`, `HybridSketchMFedMC_v2/`, `HybridSketchMFedMC_v2_flfd/`, `PMFH/` (plus
  `summarize_seeds_mhealth.py`), `FL-FD/uci_har/` (also used for mHealth via `--results_dir`/`--log_dir`)
  `FedMobile/torch_port/` and `ShapCCS/torch_port/` (needs `--prune_frac`).
- **Seed-mean plots:** each `plot_seed_mean*.py` (`CommEfficient/CommEfficient/`, `MFedMC/mHealth_v2/`,
  `PMFH/`, `FL-FD/mhealth/`, `FedMobile/torch_port/`, `ShapCCS/torch_port/`, `HybridSketchMFedMC_v2/`) writes
  a `*_mean.csv`. ShapCCS is not in `plot_mhealth_budget_comparison.py` yet.
  `plot_mhealth_budget_comparison.py` reads those and writes to `results/mhealth_budget_comparison/`.
  `--exclude FL-FD --suffix fedmobile` writes a `*_comparison_fedmobile.*` variant rather than
  overwriting. `HybridSketchMFedMC_v2/plot_configs_overlay.py` overlays all configs in one budget folder.
- **Plot style:** no std in graphs (std goes to the terminal and CSV), y-axis 0–100 with ticks every 20,
  sketch size / clients / modalities shown on the graph, legend and info box inside the axes, and a fixed
  colour per method (`COLORS` in `plot_mhealth_budget_comparison.py`; Hybrid is blue and thicker).
- **Upload budgets:** UCI HAR 10/20/30/40/50 MB, mHealth 5/10/15/20/25 MB, one graph per budget. Baselines
  use the round count nearest each budget, so some overshoot it. The user picks the Hybrid config per budget
  by hand; for the mHealth comparison:

  | Budget | Client ratio | Top modalities | Sketch | Rounds |
  |---|---|---|---|---|
  | 5 MB | 0.2 | 3 | 3×5000 | 42 |
  | 10 MB | 0.2 | 3 | 3×5000 | 84 |
  | 15 MB | 0.1 | 3 | 3×15000 | 84 |
  | 20 MB | 0.1 | 3 | 3×20000 | 84 |
  | 25 MB | 0.1 | 3 | 3×25000 | 84 |

## Methods

### B1 — IndependentCompression + FetchSGD (`CommEfficient/`)

`mm_train_independent.py` (2-mod) / `mm_train_independent_4mod.py` (mHealth) /
`mm_train_independent_4mod_v2.py` (mHealth; same method plus the compact table, tee'd `IC_` log and sweep).
Per modality: `FeaExtractor` (in_dim→512→512) + `FeaRefiner` (sigmoid gate); fusion is the element-wise
**sum** of refined features → integrator MLP → classifier. Training is original FetchSGD: every round each
selected client sketches its full gradient into one Count Sketch (`--num_rows 3 --num_cols 5000`),
server unsketches top-`k=20000` with virtual momentum + error feedback. UCI HAR config: 85 epochs,
all 10 clients/round, `lr_scale 0.05`. mHealth `_v2`: 42 rounds, `pivot_epoch 10`, with `--num_cols` varied
for the budget sweep.

General CommEfficient architecture (also used by SketchFusion A/B/C runners):
- `mm_train*.py` entry points; `fed_aggregator.py` (server: `FedModel`/`FedOptimizer`, sketch
  unrolling, virtual momentum, error feedback); `fed_worker.py` (one `torch.multiprocessing`
  process per simulated client).
- `models/` — `SketchFusionA/B/C` (A: learned hash fns, B: fixed random Count Sketch + MLP head,
  C: Tensor Sketch fusion), `independent_compression_net(_4).py`, `multimodal_net.py`, CV ResNets.
- `data_utils/` — `Fed*` dataset classes + Dirichlet `FedSampler`. `utils.py` — `parse_args`,
  `get_param_vec`/`set_param_vec`/`get_grad`.
- `--mode {sketch,true_topk,local_topk,fedavg,uncompressed}` selects upload compression.

### B2 — MFedMC v2 (`MFedMC/UCI_HAR_v2/`, `MFedMC/mHealth_v2/`)

Self-contained folder (`dataset/models/options/federated/main.py`, no CommEfficient imports).
Each modality is an independent model (IC-shaped Extractor + optional Refiner + small classifier),
trained and uploaded **uncompressed**. Each round: `compute_modality_priority()` scores modalities
by SHAP + model size + recency (`--top_shap 1` uploads one modality per client) and clients are
selected by loss (`--client_select_ratio`). A local RandomForest only feeds SHAP: the paper's RF fusion
evaluation is commented out (accuracy columns are NaN placeholders), and `test_acc` is the argmax of
summed log-probs. Aggregation is sample-count weighted; download bytes are logged. Uses `--partition
dirichlet --eval_on_global_test --train_ratio 1.0` for the shared protocol. mHealth config: `feat_dim 512`,
5 local epochs, client ratio 0.1, with rounds varied per budget. Its script header comment is out of date;
trust the flags.

PTB-XL limb/chest (`ptbxl_mm_p44`): `run_ptbxl_mfedmc_v2.sh` → `MFedMC/ptbxl/run_ptbxl_mm.sh`. `MFedMC/ptbxl/main.py`
imports UCI_HAR_v2's code unchanged (via importlib) and writes everything under `MFedMC/ptbxl/`: logs
`result_updated/MFedMC_v2_<dataset>_mm_..._seed<s>.txt`, npz `results/<dataset>/MFedMC_v2_<dataset>_mm_..._seed<s>.npz`,
summary via `MFedMC/ptbxl/summarize_seeds.py`.

`MFedMC/UCI_HAR_paper/`** — paper reproduction (see B2).
- **`Multimodal_Federated_Learning/`** — the ACM paper draft (`main.tex`, `figures/`), untracked.

## Working conventions

These come from the user's own instructions in past sessions.

- **Do not modify existing scripts or existing Hybrid files.** Add a new script or file instead;
  existing ones are referenced by prior results. Delete only when asked.
- **Baseline ports keep the algorithm.** Change only data and logging adapters. Where released code
  and the paper differ, follow the released code and document the difference.
- **Adding a baseline:** vendor the paper and upstream code, show the workflow visually, verify the
  port, and only then add runners, a seed sweep and a summarizer.
- **Paper reproductions are a separate track.** They keep upstream behaviour and logging.
- **Runner values are user state.** The user hand-edits `--rounds`, `NUM_SEEDS` and similar between
  runs, so the numbers in a script may be mid-experiment rather than canonical. Preserve them.
- **Re-read files before explaining them.** Scripts change between turns, even within a session.
- **Parallel sessions share context.** Work is often split across sibling Claude sessions; check
  their transcripts before saying something is missing.
- **Long runs:** the user starts these in their own terminal, inside tmux.
- **Commit only when asked.**

## Setup

```bash
source venv/bin/activate                 # main interpreter for CommEfficient/csh
pip install -r requirements.txt
cd csh && pip install -e . && cd ..      # csvec, editable
cd CommEfficient && pip install -e . && cd ..
```

Interpreters differ per runner — check the `PYTHON=` logic in the script before assuming:
- `MFedMC/.venv/bin/python` if present, else `python3`: MFedMC v2, MFedMC paper, Hybrid v2, FLASH port,
  FedMobile port, ShapCCS port.
- root `venv/bin/python`: PMFH, all FL-FD folders, Hybrid v2 FL-FD (`MFedMC/.venv` lacks pyts and
  Pillow), PEPSY (hardcoded, with `PYTHONPATH=PEPSY/deps`), and plotting (`MFedMC/.venv` has no
  matplotlib).
- plain `python3` (with `PYTHONPATH=CommEfficient/`): IC and other CommEfficient scripts.

## Common commands

**Prepare a dataset** (raw data → feature cache consumed by training). Raw UCI HAR is at the repo root
in `human+activity+recognition+using+smartphones/`; other raw data is under `datasets/`.
```bash
python CommEfficient/CommEfficient/prepare_uci_har_mm.py --uci_root <raw_dir> \
  --out_dir datasets/uci_har_mm                                 # both flags required
python CommEfficient/CommEfficient/prepare_mhealth_mm.py --test_split random_group \
  --out_dir datasets/mhealth_mm_random_group                    # mHealth (4 modalities)
```
Each dataset has a `prepare_*.py` in `CommEfficient/CommEfficient/`; many are kept for reference with no
cache or launcher. FL-FD builds its own UP-Fall cache (`FL-FD/fixed/prepare_upfall.py`, called
automatically by its runner).

**Run an experiment** — always via a `run_*.sh` script, not by invoking the training script
directly; the scripts set `PYTHONPATH`/interpreter and pin the full flag set for that config:
```bash
bash run_uci_har_independent_compression.sh          # B1
bash run_uci_har_mfedmc_sketchfusion_protocol_v2.sh  # B2
bash run_uci_har_hybrid_sketch_mfedmc_v2.sh          # B3 (ours)
bash PMFH/run_uci_har_pmfh_fedhead.sh                # B4
bash run_uci_har_flash.sh                            # FLASH
bash run_uci_har_flfd_features.sh                    # FL-FD on UCI HAR (feature-vector version)
bash run_upfall_flfd.sh                              # FL-FD (UP-Fall)
bash run_uci_har_fedmobile.sh                        # FedMobile: currently the _miss runner, see below
bash run_uci_har_shapccs.sh                          # ShapCCS: prune_frac 0.7 and 0.3, 5 seeds each
```
mHealth counterparts: `run_mhealth_independent_compression_4mod_v2.sh`,
`run_mhealth_mfedmc_sketchfusion_protocol_v2.sh`, `run_mhealth_hybrid_sketch_mfedmc_v2.sh`,
`PMFH/run_mhealth_pmfh_fedhead_nmod.sh`, `run_mhealth_flash.sh`, `run_mhealth_flfd_features.sh`,
`run_mhealth_fedmobile.sh`, `run_mhealth_shapccs.sh`. Other root `run_*.sh` files are older configs (CV, MIR-Flickr, SketchFusion
A/B/C, v1 MFedMC/Hybrid).

FLASH dataset (Hybrid and IC only so far; `DATA_PATH=datasets/flash_mm_episode` selects the leakage check):
```bash
venv/bin/python CommEfficient/CommEfficient/prepare_flash_mm.py --out_dir datasets/flash_mm
venv/bin/python CommEfficient/CommEfficient/prepare_flash_mm.py --test_split episode --out_dir datasets/flash_mm_episode
bash run_flash_hybrid_sketch_mfedmc_v2.sh
bash run_flash_independent_compression.sh
```
IC's rendezvous port is hardcoded (`tcp://127.0.0.1:29500` in `fed_aggregator.py` and `fed_worker.py`), so only one
CommEfficient run can be alive at a time; a second fails with `EADDRINUSE`.

Extra args are forwarded to the Python entry point, but the round flag differs: `--num_epochs` (B1, B3),
`--iterations` (B2), `--rounds` (the others). `run_mhealth_independent_compression_4mod.sh` forwards
nothing. In a sweep, a forwarded `--seed` overrides every seed in the loop.

Root-level `run_*.sh` for Hybrid v2 / FLASH / FL-FD / FedMobile are one-line `exec` wrappers around the
real script inside the method's folder — edit flags there. Note `run_uci_har_fedmobile.sh` currently
execs `FedMobile/torch_port/run_uci_har_miss.sh`, while `run_mhealth_fedmobile.sh` execs the complete-data
`run_mhealth.sh`.

**Seed sweeps.** Runners take `NUM_SEEDS` and `SEEDS` as environment variables, usually seeds 42–46:

| Behaviour with no env vars | Runners |
|---|---|
| 5 seeds | B2 and B3 on mHealth, PMFH `fedhead` and `nmod`, FL-FD `uci_har` features and `mhealth`, all FedMobile, ShapCCS (for each of `PRUNE_FRACS="0.7 0.3"`) |
| 5 seeds, values `47 48 44 45 46` | `run_mhealth_independent_compression_4mod_v2.sh` |
| Seed 42; set the env vars to sweep (B1 defaults to `NUM_SEEDS=1`, B2 and B3 to 5) | B1, B2 and B3 on UCI HAR |
| 1 seed | `PMFH/run_mhealth_pmfh_fedhead_grouped.sh`, `HybridSketchMFedMC_v2_flfd/run_uci_har.sh` |
| No sweep logic | FLASH, `FL-FD/fixed/`, other PMFH UCI runners |

A failed seed must not stop a sweep, and each sweep should end with a summary. When writing a new
runner, pass the runner's actual dataset to the summarizer: a hardcoded `--dataset` that mismatches
`DATA_PATH`, combined with `|| true`, silently produces no summary table.

**Tests:**
```bash
python -m unittest csh/csvec/test_csvec.py    # real unittest suite for the sketch library
```
`CommEfficient/CommEfficient/{grad_test,unit_test,test_gradients,pytorch_test}.py` are standalone
diagnostic/debug scripts, not a CI-style suite.

## Datasets (`datasets/`)

| Folder | Modalities (dims) | Classes | Split |
|---|---|---|---|
| `uci_har_mm` | Acc 348, Gyro 213 (keys still `img_*`/`txt_*`) | 6 | official test set, 2947 samples |
| `uci_har_lstm_mm` | `body_acc`, `body_gyro` (128×3) | 6 | official test set |
| `mhealth_mm` | Acc 177, Gyro 118, Mag 118, ECG 43 | 12 | subjects 1–8 train, 9–10 test (4188/1035) |
| `mhealth_mm_random_group` | same | 12 | 70/30 split by group (3681/1542) |
| `mhealth_mm_random` | same | 12 | 70/30 random split (3656/1567) |
| `ptbxl_reduce` | 12 ECG leads (1000×12) | 5 | folds 1–8 train, 9–10 test (3170/800); PEPSY only |
| `flash_mm` | GPS 2, Image 512 (ResNet18), LiDAR 1006 | 49 | random 90/10 within each episode (28268/3244); near-duplicate frames can cross the split |
| `flash_mm_episode` | same | 49 | all episodes numbered 9 held out (28512/3000) |

`datasets/flash_processed/` holds the FLASH zip and `features_resnet18.npz` (features before splitting, reused by
`prepare_flash_mm.py`). The zip's `rf` has 64 columns: 411 all-zero rows are dropped and the 49 sectors that occur are
remapped, so the cache has 31,512 of the 31,923 samples and differs from the paper's 34 sectors. Stored frames are BGR.
The repository blocks command-line downloads (HTTP 403), so the zip was fetched through Chrome.

`datasets/mhealth+dataset/` and `datasets/ptb-xl-*/` hold raw data.

## Shared protocol (all UCI HAR / mHealth comparisons)

- **mHealth split:** `mhealth_mm` is the main cache, but runner defaults are mixed, so check
  `DATA_PATH` before comparing numbers:
  - `mhealth_mm`: IC `_4mod_v2`, B2, B3, PMFH `nmod`, FL-FD, FedMobile `run_mhealth_miss.sh`.
  - `mhealth_mm_random_group`: IC `_4mod`, PMFH `grouped`, FLASH, FedMobile `run_mhealth.sh`.
- **Clients:** 10-client Dirichlet(α=0.1) label-skew partition, all client data is training data
  (except FLASH mHealth, which holds out `--val_frac 0.1`); evaluation on the shared test set.
- **Metric:** `test_acc` = mean accuracy over 8-sample test shards (`valid_batch_size 8`), the IC
  convention. B2, B3, FLASH, FedMobile and PMFH-head variants import/replicate this so numbers are
  comparable. Not comparable: official PMFH (kNN on hash codes) and the MFedMC paper reproduction.
- **Communication:** bytes / 1e6 reported as "MB" (some print strings still say MiB — it's MB).
  Both upload and download are counted per round; the cumulative upload MB is embedded in the log
  filename. Exception: `FL-FD/fixed/` prints true MiB, tracks no download, and has no MB in its filename.
- **Logs:** each method tees its console table to `<method>/result_updated/<prefix>_..._<cum_up>MB_
  ..._seed<N>.txt` (IC writes to `CommEfficient/result_updated/`); `.npz` histories go to
  `results/` (IC writes no `.npz`; its `results/` holds seed-mean CSVs and plots). Both are gitignored.

## Reporting and plots

- **Results:** rows go in the user's "Results" Google Sheet with columns Methods, #Rows, #Cols, #Clients,
  #Modalities, Total Rounds, Avg Test Loss, Avg Test Acc, Avg Time, Total Upload. Give rows in chat to
  paste rather than writing to the sheet. The headline accuracy is final-round shard accuracy averaged
  over 5 seeds. Avg Time excludes initialization (e.g. GAF building).
- **Logs stay minimal:** extra metrics are commented out of the table but still computed for the npz.
- **Seed summaries:** `summarize_seeds.py` in `CommEfficient/CommEfficient/`, `MFedMC/UCI_HAR_v2/`,
  `MFedMC/mHealth_v2/`, `HybridSketchMFedMC_v2/`, `HybridSketchMFedMC_v2_flfd/`, `PMFH/` (plus
  `summarize_seeds_mhealth.py`), `FL-FD/uci_har/` (also used for mHealth via `--results_dir`/`--log_dir`)
  `FedMobile/torch_port/` and `ShapCCS/torch_port/` (needs `--prune_frac`).
- **Seed-mean plots:** each `plot_seed_mean*.py` (`CommEfficient/CommEfficient/`, `MFedMC/mHealth_v2/`,
  `PMFH/`, `FL-FD/mhealth/`, `FedMobile/torch_port/`, `ShapCCS/torch_port/`, `HybridSketchMFedMC_v2/`) writes
  a `*_mean.csv`. ShapCCS is not in `plot_mhealth_budget_comparison.py` yet.
  `plot_mhealth_budget_comparison.py` reads those and writes to `results/mhealth_budget_comparison/`.
  `--exclude FL-FD --suffix fedmobile` writes a `*_comparison_fedmobile.*` variant rather than
  overwriting. `HybridSketchMFedMC_v2/plot_configs_overlay.py` overlays all configs in one budget folder.
- **Plot style:** no std in graphs (std goes to the terminal and CSV), y-axis 0–100 with ticks every 20,
  sketch size / clients / modalities shown on the graph, legend and info box inside the axes, and a fixed
  colour per method (`COLORS` in `plot_mhealth_budget_comparison.py`; Hybrid is blue and thicker).
- **Upload budgets:** UCI HAR 10/20/30/40/50 MB, mHealth 5/10/15/20/25 MB, one graph per budget. Baselines
  use the round count nearest each budget, so some overshoot it. The user picks the Hybrid config per budget
  by hand; for the mHealth comparison:

  | Budget | Client ratio | Top modalities | Sketch | Rounds |
  |---|---|---|---|---|
  | 5 MB | 0.2 | 3 | 3×5000 | 42 |
  | 10 MB | 0.2 | 3 | 3×5000 | 84 |
  | 15 MB | 0.1 | 3 | 3×15000 | 84 |
  | 20 MB | 0.1 | 3 | 3×20000 | 84 |
  | 25 MB | 0.1 | 3 | 3×25000 | 84 |

## Methods

### B1 — IndependentCompression + FetchSGD (`CommEfficient/`)

`mm_train_independent.py` (2-mod) / `mm_train_independent_4mod.py` (mHealth) /
`mm_train_independent_4mod_v2.py` (mHealth; same method plus the compact table, tee'd `IC_` log and sweep).
Per modality: `FeaExtractor` (in_dim→512→512) + `FeaRefiner` (sigmoid gate); fusion is the element-wise
**sum** of refined features → integrator MLP → classifier. Training is original FetchSGD: every round each
selected client sketches its full gradient into one Count Sketch (`--num_rows 3 --num_cols 5000`),
server unsketches top-`k=20000` with virtual momentum + error feedback. UCI HAR config: 85 epochs,
all 10 clients/round, `lr_scale 0.05`. mHealth `_v2`: 42 rounds, `pivot_epoch 10`, with `--num_cols` varied
for the budget sweep.

General CommEfficient architecture (also used by SketchFusion A/B/C runners):
- `mm_train*.py` entry points; `fed_aggregator.py` (server: `FedModel`/`FedOptimizer`, sketch
  unrolling, virtual momentum, error feedback); `fed_worker.py` (one `torch.multiprocessing`
  process per simulated client).
- `models/` — `SketchFusionA/B/C` (A: learned hash fns, B: fixed random Count Sketch + MLP head,
  C: Tensor Sketch fusion), `independent_compression_net(_4).py`, `multimodal_net.py`, CV ResNets.
- `data_utils/` — `Fed*` dataset classes + Dirichlet `FedSampler`. `utils.py` — `parse_args`,
  `get_param_vec`/`set_param_vec`/`get_grad`.
- `--mode {sketch,true_topk,local_topk,fedavg,uncompressed}` selects upload compression.

### B2 — MFedMC v2 (`MFedMC/UCI_HAR_v2/`, `MFedMC/mHealth_v2/`)

Self-contained folder (`dataset/models/options/federated/main.py`, no CommEfficient imports).
Each modality is an independent model (IC-shaped Extractor + optional Refiner + small classifier),
trained and uploaded **uncompressed**. Each round: `compute_modality_priority()` scores modalities
by SHAP + model size + recency (`--top_shap 1` uploads one modality per client) and clients are
selected by loss (`--client_select_ratio`). A local RandomForest only feeds SHAP: the paper's RF fusion
evaluation is commented out (accuracy columns are NaN placeholders), and `test_acc` is the argmax of
summed log-probs. Aggregation is sample-count weighted; download bytes are logged. Uses `--partition
dirichlet --eval_on_global_test --train_ratio 1.0` for the shared protocol. mHealth config: `feat_dim 512`,
5 local epochs, client ratio 0.1, with rounds varied per budget. Its script header comment is out of date;
trust the flags.

`MFedMC/UCI_HAR_paper/` (`run_uci_har_mfedmc_paper.sh`) reproduces arXiv:2401.16685v2 and deliberately does
not follow the shared protocol: 30 subject clients with local holdout, LSTM encoders on raw inertial signals,
and per-client RandomForest fusion accuracy. v1 (`MFedMC/UCI_HAR/`) uses an MLP encoder (`VectorMLP`).

### B3 — HybridSketchMFedMC v2, the proposed method (`HybridSketchMFedMC_v2/`)

Independent of `HybridSketchMFedMC/` (v1) and of `CommEfficient/` (Count Sketch is copied into
`sketch.py`, matching `csvecN.CSVecFed` + `_server_helper_sketched`).
- **Model** (`models.py`, `SketchFusionBNet`): per-modality `FeaExtractor` + `FeaRefiner` →
  fusion → integrator MLP → classifier, for any number of modalities. `--fusion_mode sketch`
  (default; SketchFusionB fixed Count Sketch, runners pin `--sketch_r 2 --sketch_c 128`, then sum) or `sum`
  (IC-style). No MFM / cross-modal heads.
- **Round loop** (`federated.py:run_federated`):
  1. forward-only scoring of every client on the global model → loss, acc, SHAP per modality
     (`selection.shap_from_fusion`);
  2. **client selection** (`--client_select loss` = MFedMC priority over loss, `--prefer-higher-loss`
     default on; the default weights `[1.0, 0.0]` give recency no effect; or `random`), fraction
     `--client_select_ratio` (runners: 0.4 UCI HAR, 0.1 mHealth);
  3. **modality selection** per selected client via `compute_modality_priority` (SHAP + branch size
     + recency, `--modality_weights`), keeping `--num_select_modalities` (1) branches; gradient
     coordinates of unselected branches are masked (shared fusion/head coordinates always sent);
  4. each selected client sketches its masked **gradient** (`--upload_object gradient`, default;
     `delta` sketches global−local after local SGD, only meaningful with `--local_epochs > 1`);
  5. **example-weighted** (n_i/N) sketch sum → FetchSGD server step (unsketch top-`k`, virtual
     momentum, error feedback, scheduled LR; LR 1.0 for deltas). IC-style weight decay
     `grad += (wd / n_participating) * w`.
- **Comm:** upload = `num_rows × num_cols × 4` bytes per participating client (runners pin 3×20000);
  download billed IC-style as changed coordinates since the client last synced.
- `selection.py` (priorities), `sketch.py` (CSVecFed + server step), `dataset.py`
  (`load_uci_har_mm`, `load_mhealth_mm` — FLASH imports these too), `utils.py` (`Timer`,
  `TableLogger`, param-vec helpers).
- UCI HAR config: 53 epochs, `pivot_epoch 15`, `lr_scale 0.05`, `k 20000`, `wd 5e-4`.

`HybridSketchMFedMC_v2_flfd/` runs the same algorithm with FL-FD's small CNN on 70×70 GAF images:
`main.py` imports Hybrid's unmodified code and swaps the loader and model in memory. `--model branch` has
3,342 params (against Hybrid's 2,261,510), with a 3×500 sketch and `k=500`. Experimental: one seed so far,
with noticeably longer run time.

### B4 — PMFH (`PMFH/`)

Vendored from github.com/vindahi/PMFH (Prototype-based Federated Multimodal Hashing). Upstream
files (`layers.py`, `loss.py`, `utils.py`, `*.sh` for Flickr/NUS/Wiki) are essentially as released, with small
edits to `main.py` (loss tracking) and `flickr.sh`. Acc → PMFH "image" input, Gyro → "text" input. UCI HAR
wrappers:
- `main_uci_har.py` / `run_uci_har_pmfh.sh` — **official PMFH**; accuracy is kNN (Hamming) on hash
  codes, plus mAP and a frozen linear probe (`test_acc_probe`). Upstream's `classify` head is never
  trained.
- `main_uci_har_v2.py` — variants with a trained classifier head so `test_acc` is read out like the
  other baselines:
  - default (`run_uci_har_pmfh_v2.sh`, "PMFH-v2"): `layers_uci.LayersUCI` with IC-style gate + head
    added to the global model's loss;
  - `--variant fedhead` (`run_uci_har_pmfh_fedhead.sh`, "PMFH + federated head"): upstream model,
    head on `nec_vec.detach()`, FedAvg'd with the model;
  - `--variant fedhead --head_grad joint` (`run_uci_har_pmfh_jointhead.sh`, "PMFH + jointly
    trained head"): same but CE gradient flows into PMFH.
- mHealth: `main_mhealth_pmfh.py` is the `fedhead` variant for 4 modalities. `run_mhealth_pmfh_fedhead_nmod.sh`
  (`PMFHfhN`, one encoder and gate per modality, `layers_nmod.py`) is the one used in the plots;
  `run_mhealth_pmfh_fedhead_grouped.sh` (`PMFHfhG`) maps IMU → image and ECG → text. Both pin `bit 64`,
  `common_dim 32`.
- Edge cases handled: clients with < `classes` samples are excluded (empty prototypes → NaN);
  a final batch of size 1 is dropped (BatchNorm).

### FLASH (`FLASH/`, PyTorch port in `FLASH/torch_port/`)

Upstream: github.com/batoolsalehi/FL-based-Sector-Selection @ 94cd337 (Salehi et al., INFOCOM'22;
paper `FLASH/FLASH.pdf`). Original task: 34-class mmWave beam-sector selection from GPS + camera +
LiDAR, Keras. `FLASH/code/`, `FLASH/preprocessing/` are upstream and not run here.
- `flash_model.py`: one branch per modality → concat → FLASH fusion head (Dense
  1024/512/256/128). `--branch gps` = FLASH's coord_mlp branch (`--branch_dims`), `--branch
  shared` = Hybrid v2's Extractor+Refiner. The UCI HAR runner uses `shared`, `--fusion_dims
  128,64,32,16`, `--feat_dim 64`; the mHealth runner uses the defaults (`gps`, full widths).
- `flash_federated.py`: all clients every round; each client overwrites the branch(es) its policy
  draw selects with global weights, trains with Adam (≤ `local_epochs`, early stopping patience 5 when
  `--val_frac > 0`: mHealth only, since UCI HAR sets 0.0), uploads its full model; server takes the
  **unweighted** mean (incl. BN stats). `--policy all` = released `run_flash.sh` (P_All);
  `--policy p_mod0,...,p_fusion` = paper's random branch policy. Fixes vs. released Keras: independent
  client copies and a fresh Adam per client.
- `run_uci_har_small.sh` = widths /8 variant (419k params, 1.68 MB per upload).
- Known behaviour: under Dirichlet(0.1) with short local training, averaged BN moving stats can make
  the model predict one class (~18%) — expected, not a bug.

### FL-FD (`FL-FD/`: `fixed/`, `uci_har/`, `mhealth/`)

Upstream: MODAL-UNINA Federated-Learning-based-Fall-Detection (Qi et al., Information Fusion 2023;
paper `FL-FD/FL-FD.pdf`). `FL-FD/fixed/` is the fixed port of the original setup, input-level fusion on
**UP-Fall** (it needs a camera):
[GAF(acc), GAF(gyro), camera frame-difference] → 3×70×70 → 3-layer CNN, FedAvg, 15 subject
clients (subjects 5, 9 dropped), 12 per round, trials 1–2 train / trial 3 test.
`--task activity` (11-class, 200 rounds) or `fall` (binary, 100 rounds).
- Top-level `FL-FD/main.py`, `data_processing.py` are the **released code, which is broken** (every
  client shares the server's model object so deltas are zero; dict/key bugs; 140 vs 70 px). Never run them.
- Data: `FL-FD/dataset/CompleteDataSet.csv` + `camera/` (Camera2 frames, fetched by
  `download_camera2.py` from `upfall_links.json`); both gitignored. Cache:
  `flfd_upfall_camera2_70.npz`.
- No communication-saving mechanism — full model every round.
- **Shared-split ports:** `FL-FD/uci_har/main.py` imports the algorithm from `fixed/` and Hybrid v2's loaders
  (10 clients, 8 per round, 3 local epochs, SGD). `FL-FD/mhealth/main.py` reuses it with `load_mhealth_mm`
  (4×70×70 input). UCI HAR has three runners: `run_uci_har_flfd_features.sh` (feature vector → GAF, on
  `uci_har_mm`; the canonical one), `run_uci_har_flfd.sh` (raw magnitude → GAF, on `uci_har_lstm_mm`),
  and `run_uci_har_flfd_shared.sh` (Hybrid encoder).

### FedMobile (`FedMobile/`, PyTorch port in `FedMobile/torch_port/`)

Upstream: github.com/yiliucs/WWW25 @ 41dd79c (Liu, Wang, Yuan, WWW'25, doi:10.1145/3696410.3714623),
vendored unchanged in `FedMobile/upstream/` (files in some folders, such as `USC-HAD/`, have no `.py`
extension). The port follows the **released** `USC-HAD/MMFL_main`, not the paper (they differ: the paper does
feature-level KD with a jointly trained generator, n_k·P·ΔL aggregation weights, SGD, E=5, T=100; listed in
`main.py`'s docstring).
- Model: Hybrid v2 Extractor+Refiner per modality (`--feat_dim`, default 64) → concat → one Linear;
  per-client generator G(eps, y) → latent. All 10 clients every round, Adam 1e-3, bs 16.
- Runner settings (rounds / local epochs, currently; hand-edited): `run_uci_har.sh` 16/1,
  `run_uci_har_miss.sh` 3/5, `run_mhealth.sh` 150/1, `run_mhealth_miss.sh` 3/5.
- Loss: ½CE + ½(½KL(classifier(G(y))/0.5 ‖ logits/0.5) + ½CE(classifier(G(y')), y')).
- Server: keeps clients whose local + global contribution beats last round's accuracy (all if ≤ 3), plain
  mean of parameters; generators: k-means (`--num_clusters`, default 5; the `main.py` docstring's k=3 is
  out of date) + exact Shapley over clusters.
- As released, generators are never optimised (`--train_generator 1` changes that), so after round 1
  every client holds the same generator and the k-means/Shapley step is effectively uniform.
- Two release bugs in contribution bookkeeping are fixed (documented in `main.py`).
- `run_*_miss.sh` = released missing-modality simulation (modality 0 zeroed in 80% of each batch);
  `run_uci_har.sh` and `run_mhealth.sh` use complete data like the other baselines.
- Comm: model + generator parameters up and down every round (no BN buffers; round 1 has no download).
- `summarize_seeds.py` writes the Results-sheet row to `result_updated/..._seeds<...>_summary.txt` (filters:
  `--rounds`, `--local_epochs`, `--variant`). `result_updated/` is split into `mhealth/` and `uci/`.

### ShapCCS (`ShapCCS/`, port in `ShapCCS/torch_port/`)

Ji et al., ICML'26 (paper `ShapCCS/ShapCCS.pdf`, OpenReview `PnabunIVDL`); upstream github.com/iz70778/ShapCCS
@ 8a45bbb, vendored unchanged in `ShapCCS/upstream/`. The port runs the released `fed_sv_cs_super` +
Moderate path (the paper's ShapCCS). A client-selection / coreset method with no compression: it saves
upload only by dropping clients.
- Algorithm: FedAvg (SGD 0.05·0.99^round, bs 32, E 1, 100 rounds). Rounds 0–9 pick clients round-robin
  and score them with GPSV from the change in the final `output` layer (projection on the aggregate
  change, ψ clipping, exact Shapley over coalitions). Round 10 splits the budget B = (1 − prune_frac)·N by
  GPSV, drops clients below τ = B / `num_clients_remain`, and shrinks the kept clients' data with Moderate.
  Later rounds sample only kept clients.
- `gpsv.py` is upstream's code copied with `# PORT:` lines (device moves, deepcopy, args, logging hook);
  `check_port.py` imports the upstream modules and checks equality (no dataset needed).
- Model: Hybrid v2 Extractor+Refiner per modality (`--feat_dim 64`) → concat → `output` Linear.
- Runners (both datasets): client ratio 0.2, prune_frac 0.7 with `num_clients_remain 2` and 0.3 with 7
  (released 20 and 70 of 100 clients), 5 seeds each; mHealth uses `datasets/mhealth_mm`. PTB-XL:
  `run_ptbxl_shapccs_p44.sh` → `ShapCCS/torch_port/run_ptbxl_p44.sh` on `ptbxl_mm_p44` (Limb/Chest via
  `--dataset uci_har`, as the other PTB-XL runners).
- After round 10 the clients per round are `ceil(0.2 · kept)`: 1 when ≤ 5 clients are kept, 2 when 6+, so a fixed
  `--rounds` can give different uploads per seed. `summarize_budget.py` cuts one long run per seed at the round
  nearest a cumulative-upload budget (exact, since the LR depends only on the round index).
- Kept as released: client weights are batch counts; Moderate's pruning positions come from a shuffled loader,
  so which samples it keeps is close to random (the count is right); round-robin pretraining (the paper
  samples at random).
- Comm: picked clients' full model up and down each round (≈0.33 MB per client on mHealth, 0.25 MB on UCI HAR,
  0.16 MB on PTB-XL).

### PEPSY (`PEPSY/`)

Upstream: github.com/nmduonggg/PEPSY @ 58e466a (NeurIPS'25, multimodal FL with missing data), a nested git
repo that is untracked here. A candidate baseline, not yet on the shared protocol: the upstream code runs
as-is on PTB-XL via `PEPSY/run_ptbxl.sh` (32 clients, 12 leads as modalities, 5 classes), with logs in
`PEPSY/logs/` and history JSON in `src/fedtask/<task>/record/`. `src/DATA/PTBXL_REDUCE` links to
`datasets/ptbxl_reduce`, which `CommEfficient/CommEfficient/prepare_ptbxl_reduce.py` rebuilds because upstream
did not release its preprocessing; it is a reconstruction, not a copy of their subset.

## Results & data conventions

- `.npz`/`.h5`/`.png`/`.pdf`/model weight files, `datasets/`, `PMFH/data/`, `PMFH/logs/`,
  `FL-FD/dataset/*.csv`, `FL-FD/dataset/camera/` are gitignored. Tracked files are code, apart from raw UCI HAR
  text files under `MFedMC/UCI_HAR/data/`.
- Each subproject writes `results/` and `result_updated/` (console logs), with hyperparameters encoded in the
  filename. Hybrid v2's mHealth runs are grouped into `<budget>MB/` subfolders, e.g.
  `HybridSketchMFedMC_v2/result_updated/20MB/HybridV2_mhealth_mm_mm_20.16MB_dirichlet0.1_it84_r3xc20000_k20000_lr0.05_Top_1_Client_0.1_highloss_clients10_workers1_seed42.txt`.
- Keep the filename scheme as it is unless asked, and always include the seed.
- `datasets/<name>_mm/` is the multimodal-feature variant most training scripts point `--dataset_dir` at.

## Current state (snapshot, 2026-11-03)

May go stale; check before relying on it.

- **Paper draft** in `Multimodal_Federated_Learning/main.tex` includes the UCI HAR and mHealth budget tables and
  figures. The mHealth graphs replace FL-FD with FedMobile (`--suffix fedmobile`); the original FL-FD plots
  still exist.
- **Unresolved comparability question:** FedMobile's mHealth budget results and figures come from
  `run_mhealth_miss.sh` (`E5`, `miss0r0.8`: 5 local epochs, Acc missing in 80% of batches), while the other
  methods in those comparisons use complete data.
- **FedMobile upload size:** `FEDMOBILE_ROUNDS` in `plot_mhealth_budget_comparison.py` assumes 6.21 MB per
  round (`feat_dim 64`). Runs at `feat_dim` 32 and 48 exist in its `results/` from looking at smaller uploads.
- **Baseline search:** PEPSY is being evaluated; a port to the shared protocol has been offered but not built.
  ShapCCS was ported on 2026-11-18 (`check_port.py` passes, 12-round smoke runs on both datasets); no full
  5-seed sweeps yet.
- **Stale docs:** `README.md` does not mention FLASH, FL-FD, FedMobile or PEPSY, and still names
  `mhealth_mm_random_group`. Some runner header comments also disagree with their flags.
