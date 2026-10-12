#!/usr/bin/env bash
# Eval matrix: bimodal, residual+gate, NN-T alone, NN-T + gate on held-out hard test traces.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
WARMUP="${WARMUP:-100000}"
SIM="${SIM:-200000}"
SPLIT="${SPLIT:-${ROOT}/data/splits/spec_residual_v1.json}"
RESIDUAL_MODEL="${RESIDUAL_MODEL:-${ROOT}/models/export/nn_residual_int8.bin}"
TRANSFORMER_MODEL="${TRANSFORMER_MODEL:-${ROOT}/models/export/nn_transformer.bin}"
PREDS=(bimodal anba_residual_hybrid nn_transformer_frozen anba_transformer_hybrid)

if [[ ! -f "${SPLIT}" ]]; then
  echo "missing split ${SPLIT}"
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
      echo "skip ${pred}: binary missing"
      continue
    fi
    model=""
    extra_env=""
    case "${pred}" in
      anba_residual_hybrid)
        model="${RESIDUAL_MODEL}"
        extra_env="ANBA_MODEL_PATH=${model} ANBA_NN_MARGIN=${ANBA_NN_MARGIN:-8}"
        ;;
      nn_transformer_frozen|anba_transformer_hybrid)
        model="${TRANSFORMER_MODEL}"
        extra_env="ANBA_MODEL_PATH=${model}"
        if [[ "${pred}" == "anba_transformer_hybrid" ]]; then
          margin="$(python3 -c "import json; p='${TRANSFORMER_MODEL}'.replace('.bin','.json'); print(json.load(open(p))['margin'])" 2>/dev/null || echo 1)"
          extra_env="ANBA_MODEL_PATH=${model} ANBA_NN_MARGIN=${margin}"
        fi
        ;;
      *)
        model="none"
        ;;
    esac
    echo "=== ${pred} $(basename "${tr}") model=$(basename "${model:-na}") ==="
    python3 "${ROOT}/tools/run_sim.py" --bin "${bin}" --trace "${tr}" \
      --predictor "${pred}" --warmup "${WARMUP}" --sim "${SIM}" \
      --extra-env "${extra_env}" \
      || echo "run failed for ${pred} $(basename "${tr}")"
  done
done
python3 "${ROOT}/tools/summarize_transformer_matrix.py"
