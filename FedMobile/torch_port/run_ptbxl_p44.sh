#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# PTB-XL (2 modalities: Limb/Chest leads, partition seed 44) with **FedMobile**
# (Liu, Wang, Yuan, WWW'25), PyTorch port of FedMobile/upstream/USC-HAD/MMFL_main
# next to IC / Hybrid v2 (run_ptbxl_independent_compression_p44.sh,
# HybridSketchMFedMC_v2/run_ptbxl_mm.sh).
#
#   • data: datasets/ptbxl_mm_p44 (Limb 113 + Chest 113 features, 5 superclasses);
#     its cached client*.npz split (10 clients, Dirichlet(0.1), partition seed 44)
#     and official test set, read with Hybrid v2's load_uci_har_mm (--dataset uci_har),
#     as Hybrid v2's PTB-XL runner does. The log prints the modalities as Acc/Gyro:
#     Acc = Limb, Gyro = Chest (cache keys img_*/txt_*).
#   • model: Hybrid v2 Extractor+Refiner per modality (64-D) -> concat
#     -> one Linear (released MyMMModel.classifier); generator G(eps, y)
#   • recipe (released): all clients every round, 1 local epoch, Adam 1e-3,
#     wd 1e-4, bs 16, rounds as set below (released: 150); KD from the generator, contribution-aware
#     model aggregation, k-means + Shapley generator aggregation
#   • missing modality: off (--miss_rate 0), every client holds every modality
#   • test_acc = 8-sample shard mean (as the baselines)
#   • log -> FedMobile/torch_port/result_updated/FedMobile_ptbxl_mm_p44_..._<cum_up>MB_....txt
#
# Extra args are forwarded, e.g.:
#   ./FedMobile/torch_port/run_ptbxl_p44.sh --rounds 2
#   ./FedMobile/torch_port/run_ptbxl_p44.sh --miss_rate 0.8 --miss_modality 0  # released missing (Limb)
#   ./FedMobile/torch_port/run_ptbxl_p44.sh --train_generator 1                # also step the generators
# Seeds: edit NUM_SEEDS / SEEDS below, or override per call, e.g.
#   NUM_SEEDS=1 ./FedMobile/torch_port/run_ptbxl_p44.sh             # seed 42 only
#   NUM_SEEDS=3 SEEDS="7 8 9" ./FedMobile/torch_port/run_ptbxl_p44.sh
# Do NOT pass --seed as an extra arg: it would override every seed of the sweep.
# A failed seed is skipped (the next seeds still run) and listed at the end.
# After the seeds, summarize_seeds.py prints/saves their mean in the Results-sheet
# columns: result_updated/FedMobile_ptbxl_mm_p44_mm_<up>MB_it<R>_K10_E1[_tags]_seeds<...>_summary.txt
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
NUM_SEEDS="${NUM_SEEDS:-5}"         # how many seeds to run: the first NUM_SEEDS values of SEEDS
SEEDS="${SEEDS:-42 43 44 45 46}"    # seed values, space-separated

cd "${ROOT}"
read -r -a SEED_LIST <<< "${SEEDS}"
FAILED=()
DONE=()
START="$(date +%s)"
for s in "${SEED_LIST[@]:0:${NUM_SEEDS}}"; do
  echo "=== seed ${s} ==="
  if ! "${PYTHON}" "${HERE}/main.py" \
    --dataset uci_har \
    --dataset_dir "${DATA_PATH}" \
    --num_clients 10 \
    --dirichlet_alpha 0.1 \
    --rounds 150 \
    --local_epochs 1 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --bs 16 \
    --seed "${s}" \
    --device cuda \
    "$@"; then
    echo "!!! seed ${s} failed; continuing with the next seed" >&2
    FAILED+=("${s}")
  else
    DONE+=("${s}")
  fi
done

# mean over the finished seeds, in the Results-sheet columns
if (( ${#DONE[@]} )); then
  "${PYTHON}" "${HERE}/summarize_seeds.py" --dataset "$(basename "${DATA_PATH}")" \
    --since "${START}" --seeds "${DONE[@]}" || true
fi

if (( ${#FAILED[@]} )); then
  echo "Failed seed(s): ${FAILED[*]}" >&2
  exit 1
fi
