#!/usr/bin/env bash
# Copy ANBA branch predictors into the pinned ChampSim tree.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/champsim/ChampSim/branch"
if [[ ! -d "${DEST}" ]]; then
  echo "ChampSim not pinned; run make champsim-pin" >&2
  exit 1
fi
for pred in instrumented nn_frozen anba_hybrid anba_online anba_residual_hybrid nn_transformer_frozen anba_transformer_hybrid anba_online_repeat_chooser anba_online_repeat_vote anba_gshare_online gshare_repeat; do
  mkdir -p "${DEST}/${pred}"
  cp -f "${ROOT}/predictor/${pred}/"* "${DEST}/${pred}/"
done
cp -f "${ROOT}/predictor/common/anba_int8.h" "${ROOT}/champsim/ChampSim/inc/anba_int8.h"
cp -f "${ROOT}/predictor/common/anba_transformer.h" "${ROOT}/champsim/ChampSim/inc/anba_transformer.h"
cp -f "${ROOT}/predictor/common/anba_gate_stats.h" "${ROOT}/champsim/ChampSim/inc/anba_gate_stats.h"
cp -f "${ROOT}/predictor/common/anba_online_checkpoint.h" "${ROOT}/champsim/ChampSim/inc/anba_online_checkpoint.h"
cp -f "${ROOT}/predictor/common/anba_online_epoch_stats.h" "${ROOT}/champsim/ChampSim/inc/anba_online_epoch_stats.h"
cp -f "${ROOT}/predictor/common/anba_features_ext.h" "${ROOT}/champsim/ChampSim/inc/anba_features_ext.h"
cp -f "${ROOT}/predictor/common/anba_online_checkpoint_v2.h" "${ROOT}/champsim/ChampSim/inc/anba_online_checkpoint_v2.h"
echo "installed predictors into ${DEST} and inc/anba_*.h"
