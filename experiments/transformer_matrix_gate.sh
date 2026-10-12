#!/usr/bin/env bash
# Hard test matrix: bimodal, residual+gate, NN-T+gate (100k/200k).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
WARMUP="${WARMUP:-100000}"
SIM="${SIM:-200000}"
SPLIT="${SPLIT:-${ROOT}/data/splits/spec_residual_v1.json}"
RESIDUAL_MODEL="${RESIDUAL_MODEL:-${ROOT}/models/export/nn_residual_int8.bin}"
TRANSFORMER_MODEL="${TRANSFORMER_MODEL:-${ROOT}/models/export/nn_transformer.bin}"
PREDS=(bimodal anba_residual_hybrid anba_transformer_hybrid)

mapfile -t TRACES < <(python3 -c "
import json
doc = json.load(open('${SPLIT}'))
sub = doc.get('residual_subset') or doc.get('hard_domain_subset')
for t in sub['traces']['test']:
    print('${ROOT}/data/traces/' + t)
")

for tr in "${TRACES[@]}"; do
  for pred in "${PREDS[@]}"; do
    bin="${ROOT}/champsim/ChampSim/bin/champsim_${pred}"
    extra=""
    case "${pred}" in
      anba_residual_hybrid)
        extra="ANBA_MODEL_PATH=${RESIDUAL_MODEL} ANBA_NN_MARGIN=${ANBA_NN_MARGIN:-8}"
        ;;
      anba_transformer_hybrid)
        margin="$(python3 -c "import json; print(json.load(open('${TRANSFORMER_MODEL}.json'))['margin'])" 2>/dev/null || echo 20)"
        extra="ANBA_MODEL_PATH=${TRANSFORMER_MODEL} ANBA_NN_MARGIN=${margin}"
        ;;
    esac
    echo "=== ${pred} $(basename "${tr}") ==="
    if [[ -n "${extra}" ]]; then
      python3 "${ROOT}/tools/run_sim.py" --bin "${bin}" --trace "${tr}" \
        --predictor "${pred}" --warmup "${WARMUP}" --sim "${SIM}" \
        --extra-env "${extra}"
    else
      python3 "${ROOT}/tools/run_sim.py" --bin "${bin}" --trace "${tr}" \
        --predictor "${pred}" --warmup "${WARMUP}" --sim "${SIM}"
    fi
  done
done
python3 "${ROOT}/tools/summarize_transformer_matrix.py"
