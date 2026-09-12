#!/usr/bin/env bash
# Run the implemented predictors on every downloaded small-preset trace.
# Skips missing binaries/traces instead of inventing numbers.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
WARMUP="${WARMUP:-100000}"
SIM="${SIM:-200000}"
PREDS=(bimodal instrumented nn_frozen anba_hybrid anba_online)
while IFS= read -r -d '' tr; do
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
done < <(find "${ROOT}/data/traces" -name '*.champsimtrace.xz' -print0)
python3 "${ROOT}/tools/summarize_matrix.py"
python3 "${ROOT}/tools/acceptance.py"
