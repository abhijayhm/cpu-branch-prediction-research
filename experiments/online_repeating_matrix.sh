#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
for pred in bimodal anba_residual_hybrid gshare anba_online_repeat_chooser anba_online_repeat_vote; do
  make champsim-build PREDICTOR="${pred}"
done
python3 "${ROOT}/tools/run_online_epochs.py"
python3 "${ROOT}/tools/summarize_online_repeating.py"
python3 "${ROOT}/tools/make_online_learning_chart.py"
