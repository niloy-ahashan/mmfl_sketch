#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# FLASH hybrid v2: FetchSGD engine (example-weighted gradient sketch)
# + MFedMC client/modality selection (3 modalities: GPS, Image, LiDAR).
#
# --num_rows, --num_cols, --num_epochs, --pivot_epoch and the selection flags are
# the mHealth runner's values, as a starting point: no FLASH budget is chosen yet.
#
# Extra args are forwarded to main_flash.py, e.g.:
#   ./HybridSketchMFedMC_v2/run_flash.sh --num_epochs 2 --pivot_epoch 1
#
# The leakage check uses the cache that holds out whole episodes:
#   DATA_PATH=datasets/flash_mm_episode ./HybridSketchMFedMC_v2/run_flash.sh
#
# Runs a multi-seed sweep by default and prints/saves the mean. Change it with
# NUM_SEEDS and/or SEEDS in the environment, e.g.:
#   NUM_SEEDS=3 SEEDS="42 43 44" ./HybridSketchMFedMC_v2/run_flash.sh
# Do NOT pass --seed yourself as an extra arg: "$@" is appended after
# --seed "${s}" on every iteration, and argparse keeps the last occurrence of a
# flag, so a user-supplied --seed would silently override every seed of the sweep.
# A failed seed is skipped (the remaining seeds still run) and listed at the
# end, with a final nonzero exit if anything failed.
# After the seeds, summarize_seeds_flash.py prints/saves their mean in the
# Results-sheet columns: result_updated/HybridV2_..._seeds<...>_summary.txt
#
# Requires: datasets/flash_mm (see prepare_flash_mm.py).
# Dims and num_classes come from prepare_stats.json.
# ---------------------------------------------------------------

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
DATA_PATH="${DATA_PATH:-${ROOT}/datasets/flash_mm}"

if [[ -x "${ROOT}/MFedMC/.venv/bin/python" ]]; then
  PYTHON="${ROOT}/MFedMC/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

# Flags shared by every run; --seed is appended separately below.
COMMON_ARGS=(
  --dataset_dir "${DATA_PATH}"
  --num_classes 49
  --num_clients 10
  --dirichlet_alpha 0.1
  --feat_dim 512
  --sketch_r 4
  --sketch_c 128
  --mm_dropout 0.3
  --local_epochs 1
  --local_batch_size -1
  --upload_object gradient
  --weight_decay 5e-4
  --virtual_momentum 0.9
  --error_type virtual
  --mode sketch
  --k 20000
  --num_rows 3
  --num_cols 10000
  --num_epochs 85
  --pivot_epoch 20
  --lr_scale 0.1
  --num_blocks 1
  --fusion_mode sketch
  --client_select loss
  --client_select_ratio 0.5
  --num_select_modalities 2
  --device cuda
)

cd "${ROOT}"

NUM_SEEDS="${NUM_SEEDS:-1}"         # how many seeds to run: the first NUM_SEEDS values of SEEDS
SEEDS="${SEEDS:-42 43 44 45 46}"    # seed values, space-separated

read -r -a SEED_LIST <<< "${SEEDS}"
FAILED=()
DONE=()
START="$(date +%s)"
for s in "${SEED_LIST[@]:0:${NUM_SEEDS}}"; do
  echo "=== seed ${s} ==="
  if ! "${PYTHON}" "${HERE}/main_flash.py" "${COMMON_ARGS[@]}" --seed "${s}" "$@"; then
    echo "!!! seed ${s} failed; continuing with the next seed" >&2
    FAILED+=("${s}")
  else
    DONE+=("${s}")
  fi
done

# mean over the finished seeds, in the Results-sheet columns; the log names
# carry the dataset dir's basename (main.py _txt_stem).
if (( ${#DONE[@]} )); then
  "${PYTHON}" "${HERE}/summarize_seeds_flash.py" --dataset "$(basename "${DATA_PATH}")" \
    --since "${START}" --seeds "${DONE[@]}" || true
fi

if (( ${#FAILED[@]} )); then
  echo "Failed seed(s): ${FAILED[*]}" >&2
  exit 1
fi
