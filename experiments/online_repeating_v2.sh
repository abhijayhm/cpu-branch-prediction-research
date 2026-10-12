#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
for pred in bimodal gshare_repeat hashed_perceptron anba_gshare_online anba_online_repeat_chooser anba_online_repeat_vote; do
  make champsim-build PREDICTOR="${pred}"
done
python3 -m unittest tests/test_online_vote_aggregate.py -v
python3 "${ROOT}/tools/tune_online_hyperparams.py"
python3 "${ROOT}/tools/run_online_repeating_v2.py"
python3 "${ROOT}/tools/summarize_online_repeating_v2.py"
python3 "${ROOT}/tools/make_online_repeating_charts_v2.py"
