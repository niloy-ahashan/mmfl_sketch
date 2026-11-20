#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# PTB-XL limb/chest (ptbxl_mm_p44) with **PMFH + logit-adjusted federated
# head** (PMFHla): run_ptbxl_pmfh_fedhead.sh's setup, but the head's CE uses
# CE(logits + tau*log pi_client, y) during local training (raw logits at test),
# so the FedAvg'd head no longer collapses to the majority class (NORM) under
# Dirichlet(0.1). PMFH, comm, FedAvg, data and metric are unchanged;
# --la_tau 0 reproduces PMFHfh exactly.
#
#   • --rounds 11 -> ~40.2 MB (3.65 MB/round); 3 / 6 / 8 for the 11 / 22 / 29 MB budgets
#   • log -> PMFH/result_updated/PMFHla_ptbxl_mm_p44_mm_<cum_up>MB_..._tau<t>_it<R>_..._seed<s>.txt
#   • npz -> results/ptbxl_mm_p44/PMFHla_UCI_HAR_bit<b>_h<h>_seed<s>.npz
#
# Extra args are forwarded, e.g.:
#   ./PMFH/run_ptbxl_pmfh_lahead.sh --rounds 2 --train_ep 1 --train_ep_private 1
# Seeds: NUM_SEEDS=3 SEEDS="7 8 9" ./PMFH/run_ptbxl_pmfh_lahead.sh
# (don't pass --seed as an extra arg: it would override every seed of the loop)
# A failed seed is skipped (the next seeds still run) and listed at the end.
# After the seeds, summarize_seeds_lahead.py prints/saves their mean:
#   result_updated/PMFHla_ptbxl_mm_p44_mm_<up>MB_it<R>_bit<b>_h<h>_c<c>_seeds<...>_summary.txt
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
  if ! "${PYTHON}" "${HERE}/main_pmfh_lahead.py" \
  --variant fedhead \
  --la_tau 1.0 \
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
  --rounds 11 \
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
    find "${RESULTS_DIR}" -maxdepth 1 -name 'PMFHla_UCI_HAR_bit*_h*.npz' ! -name '*_seed*' \
      -newermt "@${SEED_START}" | while read -r f; do cp "${f}" "${f%.npz}_seed${s}.npz"; done
  fi
done

# mean over the finished seeds, in the Results-sheet columns
if (( ${#DONE[@]} )); then
  "${PYTHON}" "${HERE}/summarize_seeds_lahead.py" --prefix PMFHla --dataset "$(basename "${DATA_PATH}")" --since "${START}" --seeds "${DONE[@]}" || true
fi

if (( ${#FAILED[@]} )); then
  echo "Failed seed(s): ${FAILED[*]}" >&2
  exit 1
fi
