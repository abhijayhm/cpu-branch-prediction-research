#!/usr/bin/env bash
# Eval matrix for residual specialist + safer hybrid gate on held-out hard test traces.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
WARMUP="${WARMUP:-100000}"
SIM="${SIM:-200000}"
SPLIT="${SPLIT:-${ROOT}/data/splits/spec_residual_v1.json}"
RESIDUAL_MODEL="${RESIDUAL_MODEL:-${ROOT}/models/export/nn_residual_int8.bin}"
HARD_DOMAIN_MODEL="${HARD_DOMAIN_MODEL:-${ROOT}/models/export/nn_c_int8.bin}"
PREDS=(bimodal anba_hybrid anba_online anba_residual_hybrid nn_frozen)

if [[ ! -f "${SPLIT}" ]]; then
  echo "missing split ${SPLIT}; run: python3 tools/make_residual_splits.py"
  exit 1
fi

mapfile -t TRACES < <(python3 -c "
import json
doc = json.load(open('${SPLIT}'))
sub = doc.get('residual_subset') or doc.get('hard_domain_subset')
for t in sub['traces']['test']:
    print('${ROOT}/data/traces/' + t)
")

for tr in "${TRACES[@]}"; do
  if [[ ! -f "${tr}" ]]; then
    echo "skip missing trace ${tr}"
    continue
  fi
  for pred in "${PREDS[@]}"; do
    bin="${ROOT}/champsim/ChampSim/bin/champsim_${pred}"
    if [[ ! -x "${bin}" ]]; then
      echo "skip ${pred} on ${tr}: binary missing"
      continue
    fi
    model="${HARD_DOMAIN_MODEL}"
    extra_env="ANBA_MODEL_PATH=${model}"
    if [[ "${pred}" == "anba_residual_hybrid" ]]; then
      model="${RESIDUAL_MODEL}"
      extra_env="ANBA_MODEL_PATH=${model} ANBA_NN_MARGIN=${ANBA_NN_MARGIN:-8}"
    fi
    if [[ "${pred}" == "nn_frozen" ]]; then
      model="${RESIDUAL_MODEL}"
      extra_env="ANBA_MODEL_PATH=${model}"
    fi
    echo "=== ${pred} $(basename "${tr}") model=$(basename "${model}") ==="
    python3 "${ROOT}/tools/run_sim.py" --bin "${bin}" --trace "${tr}" \
      --predictor "${pred}" --warmup "${WARMUP}" --sim "${SIM}" \
      --extra-env "${extra_env}" \
      || echo "run failed for ${pred} $(basename "${tr}") (not fabricating a result)"
  done
done
python3 "${ROOT}/tools/summarize_residual_matrix.py"
