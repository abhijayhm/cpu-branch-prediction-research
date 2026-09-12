#!/usr/bin/env bash
# Run the standard predictor matrix on hard-eval traces only.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
WARMUP="${WARMUP:-100000}"
SIM="${SIM:-200000}"
SPLIT="${SPLIT:-${ROOT}/data/splits/spec_hard_v1.json}"
PREDS=(bimodal instrumented nn_frozen anba_hybrid anba_online)

if [[ ! -f "${SPLIT}" ]]; then
  echo "missing split ${SPLIT}; run: python3 tools/make_hard_splits.py"
  exit 1
fi

mapfile -t TRACES < <(python3 -c "
import json, sys
doc = json.load(open('${SPLIT}'))
for t in doc['hard_subset']['traces']['test']:
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
    echo "=== ${pred} $(basename "${tr}") ==="
    python3 "${ROOT}/tools/run_sim.py" --bin "${bin}" --trace "${tr}" \
      --predictor "${pred}" --warmup "${WARMUP}" --sim "${SIM}" \
      --extra-env "ANBA_MODEL_PATH=${ROOT}/models/export/nn_c_int8.bin" \
      || echo "run failed for ${pred} $(basename "${tr}") (not fabricating a result)"
  done
done
python3 "${ROOT}/tools/summarize_hard_matrix.py"
