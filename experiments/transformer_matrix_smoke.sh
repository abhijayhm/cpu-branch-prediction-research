#!/usr/bin/env bash
# Fast path when nn_transformer_frozen at 100k/200k exceeds ~3 min/trace.
# Priority: bimodal vs anba_transformer_hybrid (full windows); optional frozen smoke.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
WARMUP="${WARMUP:-100000}"
SIM="${SIM:-200000}"
FROZEN_WARMUP="${FROZEN_WARMUP:-10000}"
FROZEN_SIM="${FROZEN_SIM:-20000}"
SPLIT="${SPLIT:-${ROOT}/data/splits/spec_residual_v1.json}"
TRANSFORMER_MODEL="${TRANSFORMER_MODEL:-${ROOT}/models/export/nn_transformer.bin}"

mapfile -t TRACES < <(python3 -c "
import json
doc = json.load(open('${SPLIT}'))
sub = doc.get('residual_subset') or doc.get('hard_domain_subset')
for t in sub['traces']['test']:
    print('${ROOT}/data/traces/' + t)
")

echo "=== smoke: bimodal + anba_transformer_hybrid warmup=${WARMUP} sim=${SIM} ==="
for tr in "${TRACES[@]}"; do
  for pred in bimodal anba_transformer_hybrid; do
    bin="${ROOT}/champsim/ChampSim/bin/champsim_${pred}"
    margin="$(python3 -c "import json; print(json.load(open('${TRANSFORMER_MODEL}.json'))['margin'])" 2>/dev/null || echo 8)"
    extra=""
    if [[ "${pred}" == "anba_transformer_hybrid" ]]; then
      extra="ANBA_MODEL_PATH=${TRANSFORMER_MODEL} ANBA_NN_MARGIN=${margin}"
    fi
    echo "=== ${pred} $(basename "${tr}") ==="
    python3 "${ROOT}/tools/run_sim.py" --bin "${bin}" --trace "${tr}" \
      --predictor "${pred}" --warmup "${WARMUP}" --sim "${SIM}" --extra-env "${extra}"
  done
done

echo "=== smoke: nn_transformer_frozen only (reduced window ${FROZEN_WARMUP}/${FROZEN_SIM}; not comparable to bimodal IPC) ==="
tr="${TRACES[0]}"
START=$(date +%s)
python3 "${ROOT}/tools/run_sim.py" \
  --bin "${ROOT}/champsim/ChampSim/bin/champsim_nn_transformer_frozen" \
  --trace "${tr}" --predictor nn_transformer_frozen \
  --warmup "${FROZEN_WARMUP}" --sim "${FROZEN_SIM}" \
  --extra-env "ANBA_MODEL_PATH=${TRANSFORMER_MODEL}"
echo "nn_transformer_frozen_smoke_elapsed_sec=$(( $(date +%s) - START ))"

python3 "${ROOT}/tools/summarize_transformer_matrix.py"
