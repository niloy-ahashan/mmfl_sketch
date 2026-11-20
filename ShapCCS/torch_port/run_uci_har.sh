#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# UCI HAR with **ShapCCS** (Ji et al., ICML'26), port of the released code
# (ShapCCS/upstream, fed_sv_cs_super + Moderate) next to IC / MFedMC v2 / Hybrid v2 / FedMobile.
#
#   • data: shared split (Acc 348 + Gyro 213, 6 classes,
#     10-client Dirichlet(0.1), official test set), every client sample is training data
#   • model: Hybrid v2 Extractor+Refiner per modality (64-D) -> concat -> output Linear
#   • recipe (released FMNIST settings): FedAvg, SGD lr 0.05 * 0.99^round, bs 32, 1 local epoch,
#     100 rounds, GPSV over the first 10 rounds, then client coreset selection + Moderate
#   • 0.2 of the clients per round (2 of 10 before selection)
#   • both prune fractions, tau scaled from the released 100-client values (B/20, B/70):
#       prune_frac 0.7 -> num_clients_remain 2,  prune_frac 0.3 -> num_clients_remain 7
#   • test_acc = 8-sample shard mean (as the baselines)
#   • log -> ShapCCS/torch_port/result_updated/ShapCCS_..._<cum_up>MB_....txt
#
# Extra args are forwarded, e.g.:
#   ./ShapCCS/torch_port/run_uci_har.sh --rounds 12
# Seeds / prune fractions: edit below, or override per call, e.g.
#   NUM_SEEDS=1 ./ShapCCS/torch_port/run_uci_har.sh               # seed 42 only
#   PRUNE_FRACS="0.7" ./ShapCCS/torch_port/run_uci_har.sh          # one prune fraction
# A failed seed is skipped (the next seeds still run) and listed at the end.
# After each prune fraction, summarize_seeds.py prints/saves the seed mean in the Results-sheet columns.
#
# Requires: datasets/uci_har_mm/data.npz
# ---------------------------------------------------------------

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
DATA_PATH="${DATA_PATH:-${ROOT}/datasets/uci_har_mm}"

if [[ -x "${ROOT}/MFedMC/.venv/bin/python" ]]; then
  PYTHON="${ROOT}/MFedMC/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi
NUM_SEEDS="${NUM_SEEDS:-5}"            # how many seeds to run: the first NUM_SEEDS values of SEEDS
SEEDS="${SEEDS:-43 44 45 46 48}"       # seed values, space-separated
# PRUNE_FRACS="${PRUNE_FRACS:-0.7 0.}"  # fraction of the data removed at the selection round
PRUNE_FRACS="${PRUNE_FRACS:-0.3}"  # fraction of the data removed at the selection round


remain_for() {  # tau = B / num_clients_remain (released: 20 of 100 at 0.7, 70 of 100 at 0.3)
  case "$1" in
    0.7) echo 2 ;;
    0.3) echo 7 ;;
    *) echo "No num_clients_remain set for prune_frac $1; add it to remain_for()" >&2; return 1 ;;
  esac
}

cd "${ROOT}"
read -r -a SEED_LIST <<< "${SEEDS}"
FAILED=()
for pf in ${PRUNE_FRACS}; do
  rem="$(remain_for "${pf}")"
  DONE=()
  START="$(date +%s)"
  for s in "${SEED_LIST[@]:0:${NUM_SEEDS}}"; do
    echo "=== prune_frac ${pf} (num_clients_remain ${rem}), seed ${s} ==="
    if ! "${PYTHON}" "${HERE}/main.py" \
      --dataset uci_har \
      --dataset_dir "${DATA_PATH}" \
      --num_clients 10 \
      --dirichlet_alpha 0.1 \
      --rounds 32 \
      --pre_train_round 10 \
      --client_ratio 0.2 \
      --prune_frac "${pf}" \
      --num_clients_remain "${rem}" \
      --ml_method moderate \
      --lr 0.05 \
      --bs 32 \
      --local_epochs 1 \
      --feat_dim 64 \
      --seed "${s}" \
      --device cuda \
      "$@"; then
      echo "!!! prune_frac ${pf} seed ${s} failed; continuing with the next seed" >&2
      FAILED+=("pf${pf}/seed${s}")
    else
      DONE+=("${s}")
    fi
  done

  # mean over the finished seeds, in the Results-sheet columns
  if (( ${#DONE[@]} )); then
    "${PYTHON}" "${HERE}/summarize_seeds.py" --dataset "$(basename "${DATA_PATH}")" \
      --prune_frac "${pf}" --since "${START}" --seeds "${DONE[@]}" || true
  fi
done

if (( ${#FAILED[@]} )); then
  echo "Failed run(s): ${FAILED[*]}" >&2
  exit 1
fi
