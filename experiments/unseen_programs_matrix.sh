#!/usr/bin/env bash
# Unseen-program generalization: bimodal vs residual/NN-T + gate at margins 8 and 20.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
SPLIT="${SPLIT:-${ROOT}/data/splits/spec_unseen_v1.json}"
WARMUP="${WARMUP:-100000}"
SIM="${SIM:-200000}"
RESIDUAL_MODEL="${RESIDUAL_MODEL:-${ROOT}/models/export/nn_residual_int8.bin}"
TRANSFORMER_MODEL="${TRANSFORMER_MODEL:-${ROOT}/models/export/nn_transformer.bin}"

mapfile -t TRACES < <(python3 -c "
import json
doc = json.load(open('${SPLIT}'))
sel = doc.get('selected_traces') or []
if not sel:
    raise SystemExit('no selected_traces; run tools/screen_unseen_bimodal.py first')
for t in sel:
    print('${ROOT}/data/traces/' + t)
")

for tr in "${TRACES[@]}"; do
  base="$(basename "${tr}")"
  echo "=== bimodal ${base} ==="
  python3 "${ROOT}/tools/run_sim.py" --bin "${ROOT}/champsim/ChampSim/bin/champsim_bimodal" \
    --trace "${tr}" --predictor bimodal --warmup "${WARMUP}" --sim "${SIM}"

  for margin in 8 20; do
    tag="m${margin}"
    echo "=== residual+gate ${tag} ${base} ==="
    python3 "${ROOT}/tools/run_sim.py" \
      --bin "${ROOT}/champsim/ChampSim/bin/champsim_anba_residual_hybrid" \
      --trace "${tr}" --predictor anba_residual_hybrid --warmup "${WARMUP}" --sim "${SIM}" \
      --run-tag "${tag}" \
      --extra-env "ANBA_MODEL_PATH=${RESIDUAL_MODEL} ANBA_NN_MARGIN=${margin}"

    echo "=== nn-t+gate ${tag} ${base} ==="
    python3 "${ROOT}/tools/run_sim.py" \
      --bin "${ROOT}/champsim/ChampSim/bin/champsim_anba_transformer_hybrid" \
      --trace "${tr}" --predictor anba_transformer_hybrid --warmup "${WARMUP}" --sim "${SIM}" \
      --run-tag "${tag}" \
      --extra-env "ANBA_MODEL_PATH=${TRANSFORMER_MODEL} ANBA_NN_MARGIN=${margin}"
  done
done

python3 "${ROOT}/tools/summarize_unseen_programs.py"
