#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# PTB-XL (12 modalities: one per ECG lead) with **Independent Compression**
#
# Multimodal fusion is ordinary summation of refined features:
#     fused = sum(f'_I, f'_II, ..., f'_V6)
#
# The fused model is then trained with original FetchSGD
# (Rothchild et al.): one Count Sketch of the full gradient.
#
# Same records, labels, train/test split, client split and training flags as
# run_ptbxl_independent_compression.sh (2 modalities), so the two can be compared.
#
# Extra args are forwarded, e.g.:
#   ./run_ptbxl_independent_compression_12mod.sh --num_epochs 2 --pivot_epoch 1 --device cpu
#
# Runs a multi-seed sweep by default and prints/saves the mean. Change it with
# NUM_SEEDS and/or SEEDS in the environment, e.g.:
#   NUM_SEEDS=3 SEEDS="42 43 44" ./run_ptbxl_independent_compression_12mod.sh
# Do NOT pass --seed yourself as an extra arg: "$@" is appended after
# --seed "${s}" on every iteration, and argparse keeps the last occurrence of a
# flag, so a user-supplied --seed would silently override every seed of the sweep.
# A failed seed is skipped (the remaining seeds still run) and listed at the
# end, with a final nonzero exit if anything failed.
# After the seeds, summarize_seeds.py prints/saves their mean in the
# Results-sheet columns: CommEfficient/result_updated/IC_ptbxl_12mod_mm_..._seeds<...>_summary.txt
#
# Requires: datasets/ptbxl_12mod_mm (see prepare_ptbxl_12mod_mm.py).
# Dims and num_classes come from prepare_stats.json, not CLI flags.
# ---------------------------------------------------------------

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/CommEfficient${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON="${PYTHON:-python3}"

if [[ ! -f datasets/ptbxl_12mod_mm/stats.json ]]; then
  echo "Missing datasets/ptbxl_12mod_mm/stats.json — build the cache first:" >&2
  echo "  python3 CommEfficient/CommEfficient/prepare_ptbxl_12mod_mm.py --ptbxl_root datasets/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3" >&2
  exit 1
fi

# Flags shared by every run; --seed is appended separately below.
COMMON_ARGS=(
  CommEfficient/CommEfficient/mm_train_independent_nmod.py
  --dataset_dir datasets/ptbxl_12mod_mm/
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
  --num_epochs 85
  --pivot_epoch 20
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
    --dataset ptbxl_12mod_mm --since "${START}" --seeds "${DONE[@]}" || true
fi

if (( ${#FAILED[@]} )); then
  echo "Failed seed(s): ${FAILED[*]}" >&2
  exit 1
fi
