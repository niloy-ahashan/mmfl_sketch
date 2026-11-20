#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# Caltech101 (3 modalities: HOG, GIST, LBP) with **Independent Compression**
#
# Multimodal fusion is ordinary summation of refined features:
#     fused = sum(f'_HOG, f'_GIST, f'_LBP)
#
# The fused model is then trained with original FetchSGD
# (Rothchild et al.): one Count Sketch of the full gradient.
#
# Same records, labels, train/test split and client split as
# HybridSketchMFedMC_v2/run_caltech101.sh, so the two can be compared.
# All flag values are the FLASH runner's, as a starting point (30 MB upload over
# 50 rounds, equal to HybridSketchMFedMC_v2/run_caltech101.sh): no budget is chosen yet.
#
# Extra args are forwarded, e.g.:
#   ./run_caltech101_independent_compression.sh --num_epochs 2 --pivot_epoch 1 --device cpu
#
#
# Runs a multi-seed sweep by default and prints/saves the mean. Change it with
# NUM_SEEDS and/or SEEDS in the environment, e.g.:
#   NUM_SEEDS=3 SEEDS="42 43 44" ./run_caltech101_independent_compression.sh
# Do NOT pass --seed yourself as an extra arg: "$@" is appended after
# --seed "${s}" on every iteration, and argparse keeps the last occurrence of a
# flag, so a user-supplied --seed would silently override every seed of the sweep.
# A failed seed is skipped (the remaining seeds still run) and listed at the
# end, with a final nonzero exit if anything failed.
# After the seeds, summarize_seeds.py prints/saves their mean in the
# Results-sheet columns: CommEfficient/result_updated/IC_caltech101_mm_..._seeds<...>_summary.txt
#
# Requires: datasets/caltech101_mm (see prepare_caltech101_mm.py).
# Dims and num_classes come from prepare_stats.json, not CLI flags.
# ---------------------------------------------------------------

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/CommEfficient${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON="${PYTHON:-python3}"
DATA_PATH="${DATA_PATH:-datasets/caltech101_mm}"

if [[ ! -f "${DATA_PATH}/stats.json" ]]; then
  echo "Missing ${DATA_PATH}/stats.json — build the cache first:" >&2
  echo "  venv/bin/python CommEfficient/CommEfficient/prepare_caltech101_mm.py --out_dir ${DATA_PATH}" >&2
  exit 1
fi

# Flags shared by every run; --seed is appended separately below.
COMMON_ARGS=(
  CommEfficient/CommEfficient/mm_train_independent_nmod.py
  --dataset_dir "${DATA_PATH}/"
  --dataset_name MultiModal
  --model IndependentCompressionN
  --feat_dim 512
  --mm_dropout 0.3
  --dirichlet_alpha 0.1
  --local_batch_size -1
  --local_momentum 0.0
  --virtual_momentum 0.9
  --error_type virtual
  --share_ps_gpu
  --mode sketch
  --k 20000
  --num_rows 3
  --num_cols 5000
  --num_epochs 50 
  --pivot_epoch 15
  --mm_local_epochs 1
  --num_clients 10
  --num_workers 10
  --lr_scale 0.05
  --num_blocks 1
  --num_devices 1
  --device cuda
)

NUM_SEEDS="${NUM_SEEDS:-1}"         # how many seeds to run: the first NUM_SEEDS values of SEEDS
SEEDS="${SEEDS:-42 43 44 45 46}"    # seed values, space-separated

read -r -a SEED_LIST <<< "${SEEDS}"
FAILED=()
DONE=()
START="$(date +%s)"
for s in "${SEED_LIST[@]:0:${NUM_SEEDS}}"; do
  echo "=== seed ${s} ==="
  if ! "${PYTHON}" "${COMMON_ARGS[@]}" --seed "${s}" "$@"; then
    echo "!!! seed ${s} failed; continuing with the next seed" >&2
    FAILED+=("${s}")
  else
    DONE+=("${s}")
  fi
done

# mean over the finished seeds, in the Results-sheet columns
if (( ${#DONE[@]} )); then
  "${PYTHON}" "${ROOT}/CommEfficient/CommEfficient/summarize_seeds.py" \
    --dataset "$(basename "${DATA_PATH}")" --since "${START}" --seeds "${DONE[@]}" || true
fi

if (( ${#FAILED[@]} )); then
  echo "Failed seed(s): ${FAILED[*]}" >&2
  exit 1
fi
