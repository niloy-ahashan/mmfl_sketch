#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# PTB-XL (2 modalities: Limb/Chest leads, partition seed 44) with **ShapCCS** (Ji et al., ICML'26),
# port of the released code (ShapCCS/upstream, fed_sv_cs_super + Moderate) next to IC / MFedMC v2 /
# Hybrid v2 / PMFH / FedMobile on the same cache.
#
#   • data: datasets/ptbxl_mm_p44 (Limb 113 + Chest 113 features, 5 superclasses);
#     its cached client*.npz split (10 clients, Dirichlet(0.1), partition seed 44, the same for every run
#     seed) and official test set, read with Hybrid v2's load_uci_har_mm (--dataset uci_har), as the
#     other PTB-XL runners do. The log prints the modalities as Acc/Gyro: Acc = Limb, Gyro = Chest.
#   • model: Hybrid v2 Extractor+Refiner per modality (64-D) -> concat -> output Linear
#   • recipe (released FMNIST settings): FedAvg, SGD lr 0.05 * 0.99^round, bs 32, 1 local epoch,
#     GPSV over the first 10 rounds, then client coreset selection + Moderate
#   • 0.2 of the clients per round: 2 of 10 in rounds 0-9; after selection ceil(0.2 * kept),
#     i.e. 1 per round when 5 or fewer clients are kept, 2 when 6 or more
#   • both prune fractions, tau scaled from the released 100-client values (B/20, B/70):
#       prune_frac 0.7 -> num_clients_remain 2,  prune_frac 0.3 -> num_clients_remain 7
#   • test_acc = 8-sample shard mean (as the baselines)
#   • log -> ShapCCS/torch_port/result_updated/ShapCCS_ptbxl_mm_p44_mm_<cum_up>MB_....txt
#     npz -> ShapCCS/torch_port/results/ShapCCS_ptbxl_mm_p44_K10_C0.2_pf<p>_rem<k>_R<R>_seed<s>.npz
#
# Extra args are forwarded, e.g.:
#   ./ShapCCS/torch_port/run_ptbxl_p44.sh --rounds 12
# Seeds / prune fractions: edit below, or override per call, e.g.
#   NUM_SEEDS=1 ./ShapCCS/torch_port/run_ptbxl_p44.sh               # seed 42 only
#   PRUNE_FRACS="0.7" ./ShapCCS/torch_port/run_ptbxl_p44.sh          # one prune fraction
# Do NOT pass --seed as an extra arg: it would override every seed of the sweep.
# A failed seed is skipped (the next seeds still run) and listed at the end.
# After each prune fraction, summarize_seeds.py prints/saves the seed mean in the Results-sheet columns.
# For a fixed upload budget, run long and cut with summarize_budget.py --dataset ptbxl_mm_p44.
#
# Requires: datasets/ptbxl_mm_p44/{data,test,client0..9}.npz + stats.json
#   (prepare_ptbxl_mm.py, then prepare_ptbxl_clients.py --start_seed 44)
# ---------------------------------------------------------------

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
DATA_PATH="${DATA_PATH:-${ROOT}/datasets/ptbxl_mm_p44}"

if [[ ! -f "${DATA_PATH}/stats.json" ]]; then
  echo "Missing ${DATA_PATH}/stats.json — build the client cache first:" >&2
  echo "  python3 CommEfficient/CommEfficient/prepare_ptbxl_clients.py --data_dir datasets/ptbxl_mm_p44 --start_seed 44" >&2
  exit 1
fi

if [[ -x "${ROOT}/MFedMC/.venv/bin/python" ]]; then
  PYTHON="${ROOT}/MFedMC/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi
NUM_SEEDS="${NUM_SEEDS:-5}"            # how many seeds to run: the first NUM_SEEDS values of SEEDS
SEEDS="${SEEDS:-47 43 44 45 46}"       # seed values, space-separated
# PRUNE_FRACS="${PRUNE_FRACS:-0.7 0.3}"  # fraction of the data removed at the selection round
PRUNE_FRACS="${PRUNE_FRACS:-0.7}"  # fraction of the data removed at the selection round


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
      --rounds 115 \
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
