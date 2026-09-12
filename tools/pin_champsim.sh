#!/usr/bin/env bash
# Clone official ChampSim and checkout the commit in software_versions.json.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
META="${ROOT}/data/metadata/software_versions.json"
DEST="${ROOT}/champsim/ChampSim"
URL="$(python3 -c "import json; print(json.load(open('${META}'))['champsim']['url'])")"
COMMIT="$(python3 -c "import json; print(json.load(open('${META}'))['champsim']['commit'])")"

mkdir -p "${ROOT}/champsim"
if [[ ! -d "${DEST}/.git" ]]; then
  echo "cloning ${URL} -> ${DEST}"
  git clone "${URL}" "${DEST}"
fi

git -C "${DEST}" fetch --tags origin
git -C "${DEST}" checkout --force "${COMMIT}"
git -C "${DEST}" submodule update --init --recursive

GOT="$(git -C "${DEST}" rev-parse HEAD)"
if [[ "${GOT}" != "${COMMIT}" ]]; then
  echo "pin mismatch: got ${GOT} expected ${COMMIT}" >&2
  exit 1
fi

echo "ChampSim pinned at ${GOT}"
git -C "${DEST}" log -1 --oneline
