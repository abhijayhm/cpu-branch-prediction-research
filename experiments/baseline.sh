#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
PRED="${1:-bimodal}"
make champsim-build PREDICTOR="${PRED}"
make baseline PREDICTOR="${PRED}" TRACE="${TRACE:-${ROOT}/data/traces/649.fotonik3d_s-1B.champsimtrace.xz}"
