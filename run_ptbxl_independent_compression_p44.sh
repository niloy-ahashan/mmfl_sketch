#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# PTB-XL (2 modalities: Limb/Chest leads, partition seed 44) with **Independent Compression**
#
# Multimodal fusion is ordinary summation of refined features:
#     fused = f'_limb + f'_chest
# The training log calls them f'_acc and f'_gyro: Limb is the "acc" branch and
# Chest the "gyro" branch of mm_train_independent.py.
#
# The fused model is then trained with original FetchSGD
# (Rothchild et al.): one Count Sketch of the full gradient.
#
# Extra args are forwarded, e.g.:
#   ./run_ptbxl_independent_compression_p44.sh --num_epochs 2 --pivot_epoch 1 --device cpu
#
# Runs a multi-seed sweep by default and prints/saves the mean. Change it with
# NUM_SEEDS and/or SEEDS in the environment, e.g.:
#   NUM_SEEDS=3 SEEDS="42 43 44" ./run_ptbxl_independent_compression_p44.sh
# Do NOT pass --seed yourself as an extra arg: "$@" is appended after
# --seed "${s}" on every iteration, and argparse keeps the last occurrence of a
# flag, so a user-supplied --seed would silently override every seed of the sweep.
# A failed seed is skipped (the remaining seeds still run) and listed at the
# end, with a final nonzero exit if anything failed.
# After the seeds, summarize_seeds.py prints/saves their mean in the
# Results-sheet columns: CommEfficient/result_updated/IC_ptbxl_mm_p44_..._seeds<...>_summary.txt
#
# Requires: datasets/ptbxl_mm_p44/data.npz (see prepare_ptbxl_mm.py) and its client
# cache (see prepare_ptbxl_clients.py). The default FedMultiModal split gives
# client 2 no samples, and then every round is skipped, so this script stops
# rather than letting FedMultiModal build that split.
# Cache uses img_*/txt_* keys; they are limb/chest features.
# ---------------------------------------------------------------

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/CommEfficient${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON="${PYTHON:-python3}"

if [[ ! -f datasets/ptbxl_mm_p44/stats.json ]]; then
  echo "Missing datasets/ptbxl_mm_p44/stats.json — build the client cache first:" >&2
  echo "  python3 CommEfficient/CommEfficient/prepare_ptbxl_clients.py --data_dir datasets/ptbxl_mm_p44 --start_seed 44" >&2
  exit 1
fi

# Flags shared by every run; --seed is appended separately below.
COMMON_ARGS=(
  CommEfficient/CommEfficient/mm_train_independent.py
  --dataset_dir datasets/ptbxl_mm_p44/
  --dataset_name MultiModal
  --model IndependentCompression
  --acc_dim 113
  --gyro_dim 113
  --feat_dim 256
  --mm_dropout 0.3
  --num_classes 5
  --dirichlet_alpha 0.1
  --local_batch_size -1
  --local_momentum 0.0
  --virtual_momentum 0.9
  --error_type virtual
  --share_ps_gpu
  --mode sketch
  --k 20000
  --num_rows 3
  --num_cols 2000
  --num_epochs 42
  --pivot_epoch 10
  --mm_local_epochs 1
  --num_clients 10
  --num_workers 10
  --lr_scale 0.1
  --num_blocks 1
  --num_devices 1
  --device cuda
)

NUM_SEEDS="${NUM_SEEDS:-5}"         # how many seeds to run: the first NUM_SEEDS values of SEEDS
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
    --dataset ptbxl_mm_p44 --since "${START}" --seeds "${DONE[@]}" || true
fi

if (( ${#FAILED[@]} )); then
  echo "Failed seed(s): ${FAILED[*]}" >&2
  exit 1
fi
