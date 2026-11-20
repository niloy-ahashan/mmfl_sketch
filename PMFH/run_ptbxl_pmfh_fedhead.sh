#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# PTB-XL limb/chest (ptbxl_mm_p44) with **PMFH + federated head**: same
# setup as run_uci_har_pmfh_fedhead.sh (official PMFH training + Linear
# head on nec_vec.detach(), CE-trained, FedAvg'd with the global model;
# test_acc = head argmax, 8-sample shard mean), on the split B1-B3 use
# (10-client Dirichlet 0.1 cache, partition seed 44, 5 classes).
#
#   • limb leads (113-D) -> PMFH "image", chest leads (113-D) -> "text";
#     ptbxl_mm_p44 has uci_har_mm's format, so main_uci_har_v2.py runs as is
#   • --rounds 11 -> ~40.3 MB cumulative upload (3.66 MB/round: 10 clients
#     x full model + prototypes), the budget of the B1-B3 PTB-XL runs
#   • main_uci_har_v2.py names its npz "UCI_HAR", so --results_dir is
#     results/ptbxl_mm_p44 to keep it apart from the UCI HAR npz
#   • log -> PMFH/result_updated/PMFHfh_ptbxl_mm_p44_mm_<cum_up>MB_....txt
#
# Extra args are forwarded, e.g.:
#   ./PMFH/run_ptbxl_pmfh_fedhead.sh --rounds 2 --train_ep 1 --train_ep_private 1
# Seeds: NUM_SEEDS=3 SEEDS="7 8 9" ./PMFH/run_ptbxl_pmfh_fedhead.sh
# (don't pass --seed as an extra arg: it would override every seed of the loop)
# A failed seed is skipped (the next seeds still run) and listed at the end.
# Each seed's npz is also copied to results/ptbxl_mm_p44/PMFHfh_UCI_HAR_bit<b>_h<h>_seed<s>.npz.
# After the seeds, summarize_seeds.py prints/saves their mean:
#   result_updated/PMFHfh_ptbxl_mm_p44_mm_<up>MB_it<R>_bit<b>_h<h>_c<c>_seeds<...>_summary.txt
#
# Requires: datasets/ptbxl_mm_p44 (prepare_ptbxl_mm.py + prepare_ptbxl_clients.py --start_seed 44)
# ---------------------------------------------------------------

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
DATA_PATH="${DATA_PATH:-${ROOT}/datasets/ptbxl_mm_p44}"
RESULTS_DIR="${HERE}/results/$(basename "${DATA_PATH}")"

if [[ -x "${ROOT}/venv/bin/python" ]]; then
  PYTHON="${ROOT}/venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

NUM_SEEDS="${NUM_SEEDS:-5}"         # how many seeds to run: the first NUM_SEEDS values of SEEDS
SEEDS="${SEEDS:-42 43 44 45 46}"    # seed values, space-separated

cd "${HERE}"
read -r -a SEED_LIST <<< "${SEEDS}"
FAILED=()
DONE=()
START="$(date +%s)"
for s in "${SEED_LIST[@]:0:${NUM_SEEDS}}"; do
  echo "=== seed ${s} ==="
  SEED_START="$(date +%s)"
  if ! "${PYTHON}" "${HERE}/main_uci_har_v2.py" \
  --variant fedhead \
  --clf_weight 1.0 \
  --head_lr 1e-3 \
  --dataset_dir "${DATA_PATH}" \
  --num_clients 10 \
  --dirichlet_alpha 0.1 \
  --classes 5 \
  --image_dim 113 \
  --text_dim 113 \
  --bit 64 \
  --hidden_dim 128 \
  --common_dim 32 \
  --rounds 3 \
  --batch_size 64 \
  --train_ep 10 \
  --train_ep_private 10 \
  --seed "${s}" \
  --device cuda \
  --results_dir "${RESULTS_DIR}" \
  "$@"; then
    echo "!!! seed ${s} failed; continuing with the next seed" >&2
    FAILED+=("${s}")
  else
    DONE+=("${s}")
    # the runner's npz name has no seed (next seed would overwrite it): keep a per-seed copy
    find "${RESULTS_DIR}" -maxdepth 1 -name 'PMFHfh_UCI_HAR_bit*_h*.npz' ! -name '*_seed*' \
      -newermt "@${SEED_START}" | while read -r f; do cp "${f}" "${f%.npz}_seed${s}.npz"; done
  fi
done

# mean over the finished seeds, in the Results-sheet columns
if (( ${#DONE[@]} )); then
  "${PYTHON}" "${HERE}/summarize_seeds.py" --prefix PMFHfh --dataset "$(basename "${DATA_PATH}")" --since "${START}" --seeds "${DONE[@]}" || true
fi

if (( ${#FAILED[@]} )); then
  echo "Failed seed(s): ${FAILED[*]}" >&2
  exit 1
fi
