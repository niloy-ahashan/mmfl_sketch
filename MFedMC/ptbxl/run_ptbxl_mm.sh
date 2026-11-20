#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# MFedMC v2 (B2) on PTB-XL, 2 modalities: Limb leads ("Acc") and Chest leads ("Gyro"),
# 113-D features each, 5 superclasses (NORM/MI/STTC/CD/HYP).
#
# Same algorithm as the UCI HAR B2 runner: MFedMC/ptbxl/main.py imports and runs
# MFedMC/UCI_HAR_v2's code unchanged; all logs/npz are written under MFedMC/ptbxl/.
# Each modality is its own Extractor+Refiner+classifier,
# trained 5 local epochs and uploaded uncompressed; SHAP/size/recency modality priority
# (top_shap 1) and highest-loss client selection. test_acc = argmax of summed per-modality
# log-probs on the shared PTB-XL test set (folds 9-10), mean over 8-sample shards.
#
#   • clients = the cached 10-client Dirichlet(0.1) partition in DATA_PATH (partition
#     seed 44 for ptbxl_mm_p44), the same split as B1/B3
#   • feat_dim 256 -> 227,845 params = 0.91 MB per modality upload
#   • client_select_ratio 0.2 (2 clients/round) x top_shap 1 x 22 rounds = 40.10 MB upload
#     (B1/B3 on ptbxl_mm_p44: 40.80 MB)
#
# Extra args are forwarded to main.py; the round flag is --iterations, e.g.:
#   ./run_ptbxl_mfedmc_v2.sh --iterations 2
#
# Runs seeds 42-46 by default. Change it with NUM_SEEDS and/or SEEDS, e.g.:
#   NUM_SEEDS=3 SEEDS="42 43 44" ./run_ptbxl_mfedmc_v2.sh
# Do NOT pass --seed yourself as an extra arg: "$@" is appended after --seed "${s}"
# on every iteration, and argparse keeps the last occurrence of a flag, so a
# user-supplied --seed would silently override every seed of the sweep.
# A failed seed is skipped (the remaining seeds still run) and listed at the end,
# with a final nonzero exit if anything failed. Each seed's npz is saved as
# MFedMC/ptbxl/results/<dataset>/MFedMC_v2_<dataset>_mm_..._seed<s>.npz.
# After the seeds, summarize_seeds.py prints/saves their mean in the
# Results-sheet columns:
#   MFedMC/ptbxl/result_updated/MFedMC_v2_<dataset>_mm_..._seeds<...>_summary.txt
#
# Requires: DATA_PATH/data.npz (prepare_ptbxl_mm.py) and its client cache
# (prepare_ptbxl_clients.py).
# ---------------------------------------------------------------

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
DATA_PATH="${DATA_PATH:-${ROOT}/datasets/ptbxl_mm_p44}"
DATASET="$(basename "${DATA_PATH}")"

if [[ ! -f "${DATA_PATH}/stats.json" || ! -f "${DATA_PATH}/test.npz" ]]; then
  echo "Missing ${DATA_PATH}/stats.json or test.npz — build the client cache first:" >&2
  echo "  python3 CommEfficient/CommEfficient/prepare_ptbxl_clients.py --data_dir ${DATA_PATH} --start_seed 44" >&2
  exit 1
fi

if [[ -x "${ROOT}/MFedMC/.venv/bin/python" ]]; then
  PYTHON="${ROOT}/MFedMC/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

RESULTS_DIR="${HERE}/results/${DATASET}"

# Flags shared by every run; --seed is appended separately below.
COMMON_ARGS=(
  --dataset_dir "${DATA_PATH}"
  --results_dir "${RESULTS_DIR}"
  --partition dirichlet
  --num_clients 10
  --dirichlet_alpha 0.1
  --num_classes 5
  --acc_dim 113
  --gyro_dim 113
  --feat_dim 256
  --use_refiner
  --batch_size 32
  --iterations 6
  --local_epochs 5
  --client_select_ratio 0.2
  --top_shap 1
  --train_ratio 1.0
  --eval_on_global_test
)

cd "${HERE}"

NUM_SEEDS="${NUM_SEEDS:-5}"         # how many seeds to run: the first NUM_SEEDS values of SEEDS
SEEDS="${SEEDS:-42 43 44 45 46}"    # seed values, space-separated

read -r -a SEED_LIST <<< "${SEEDS}"
FAILED=()
DONE=()
START="$(date +%s)"
for s in "${SEED_LIST[@]:0:${NUM_SEEDS}}"; do
  echo "=== seed ${s} ==="
  if ! "${PYTHON}" main.py "${COMMON_ARGS[@]}" --seed "${s}" "$@"; then
    echo "!!! seed ${s} failed; continuing with the next seed" >&2
    FAILED+=("${s}")
  else
    DONE+=("${s}")
  fi
done

# mean over the finished seeds, in the Results-sheet columns
if (( ${#DONE[@]} )); then
  "${PYTHON}" "${HERE}/summarize_seeds.py" --dataset "${DATASET}" \
    --since "${START}" --seeds "${DONE[@]}" || true
fi

if (( ${#FAILED[@]} )); then
  echo "Failed seed(s): ${FAILED[*]}" >&2
  exit 1
fi
