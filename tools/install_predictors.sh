#!/usr/bin/env bash
# Copy ANBA branch predictors into the pinned ChampSim tree.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/champsim/ChampSim/branch"
if [[ ! -d "${DEST}" ]]; then
  echo "ChampSim not pinned; run make champsim-pin" >&2
  exit 1
fi
for pred in instrumented nn_frozen anba_hybrid anba_online; do
  mkdir -p "${DEST}/${pred}"
  cp -f "${ROOT}/predictor/${pred}/"* "${DEST}/${pred}/"
done
cp -f "${ROOT}/predictor/common/anba_int8.h" "${ROOT}/champsim/ChampSim/inc/anba_int8.h"
echo "installed predictors into ${DEST} and inc/anba_int8.h"
