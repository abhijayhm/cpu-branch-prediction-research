#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
make catalog
make traces
make splits
make champsim-build PREDICTOR=bimodal
make baseline
make accept
