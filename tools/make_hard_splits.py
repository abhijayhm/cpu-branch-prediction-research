#!/usr/bin/env python3
"""Freeze workload-level split for hard-workload evaluation (spec_hard_v1).

Hard test workloads are chosen from bimodal probe results (acc clearly below
~90%). Train/val use easy workloads only — never train on hard test traces.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "metadata" / "traces.csv"
OUT = ROOT / "data" / "splits" / "spec_hard_v1.json"
SEED = 20260913

# Bimodal probe (WARMUP=1e5 SIM=2e5): acc < 90%, diverse domains.
HARD_TEST_WORKLOADS = (
    "654.roms",
    "648.exchange2",
    "603.bwaves",
    "607.cactuBSSN",
)

# Easy workloads (bimodal acc >= ~95% on probe); used for train/val only.
TRAIN_WORKLOADS = (
    "649.fotonik3d",
    "638.imagick",
    "644.nab",
)
VAL_WORKLOADS = ("631.deepsjeng",)

# One simpoint per hard workload (smallest or probe-selected).
HARD_TRACE_PICK = {
    "654.roms": "654.roms_s-1021B.champsimtrace.xz",
    "648.exchange2": "648.exchange2_s-1699B.champsimtrace.xz",
    "603.bwaves": "603.bwaves_s-3699B.champsimtrace.xz",
    "607.cactuBSSN": "607.cactuBSSN_s-4248B.champsimtrace.xz",
}

TRAIN_TRACE_PICK = {
    "649.fotonik3d": "649.fotonik3d_s-1B.champsimtrace.xz",
    "638.imagick": "638.imagick_s-4128B.champsimtrace.xz",
    "644.nab": "644.nab_s-12521B.champsimtrace.xz",
}

VAL_TRACE_PICK = {
    "631.deepsjeng": "631.deepsjeng_s-928B.champsimtrace.xz",
}


def workload_key(filename: str) -> str:
    if "_s-" in filename:
        return filename.split("_s-")[0]
    return filename.split(".")[0]


def expand_workloads(by_wl: dict[str, list[str]], wls: tuple[str, ...]) -> list[str]:
    out: list[str] = []
    for wl in wls:
        out.extend(sorted(by_wl.get(wl, [])))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if OUT.exists() and not args.force:
        print(f"{OUT} already frozen; use --force to regenerate")
        return 0

    if not META.exists():
        raise SystemExit(f"missing {META}; run tools/download_traces.py --catalog-only first")

    with META.open(newline="") as f:
        rows = list(csv.DictReader(f))
    by_wl: dict[str, list[str]] = {}
    for row in rows:
        wl = row.get("workload") or workload_key(row["filename"])
        by_wl.setdefault(wl, []).append(row["filename"])

    for wl in HARD_TEST_WORKLOADS + TRAIN_WORKLOADS + VAL_WORKLOADS:
        if wl not in by_wl:
            raise SystemExit(f"workload {wl} missing from catalog")

    hard_eval_traces = [HARD_TRACE_PICK[wl] for wl in HARD_TEST_WORKLOADS]
    train_traces = [TRAIN_TRACE_PICK[wl] for wl in TRAIN_WORKLOADS]
    val_traces = [VAL_TRACE_PICK[wl] for wl in VAL_WORKLOADS]

    doc = {
        "name": "spec_hard_v1",
        "seed": SEED,
        "unit": "workload",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_catalog": "data/metadata/traces.csv",
        "source_doi": "10.5281/zenodo.10960004",
        "parent_split": "data/splits/spec_v1.json",
        "rules": [
            "Split is workload-level (SPEC benchmark), not instruction-level.",
            "All simpoints of a workload share the same split.",
            "Hard test workloads chosen from bimodal probe (acc < ~90%).",
            "Never train on hard test workloads.",
            "Train on easy workloads only (bimodal acc >= ~95% on probe).",
        ],
        "bimodal_probe": {
            "warmup": 100000,
            "sim": 200000,
            "easy_reference": "649.fotonik3d_s-1B.champsimtrace.xz",
            "easy_bimodal_acc_pct": 95.22,
        },
        "hard_subset": {
            "workloads_considered": {
                "test": list(HARD_TEST_WORKLOADS),
                "train": list(TRAIN_WORKLOADS),
                "val": list(VAL_WORKLOADS),
            },
            "workloads": {
                "train": list(TRAIN_WORKLOADS),
                "val": list(VAL_WORKLOADS),
                "test": list(HARD_TEST_WORKLOADS),
            },
            "traces": {
                "train": train_traces,
                "val": val_traces,
                "test": hard_eval_traces,
            },
            "full_workload_traces": {
                "train": expand_workloads(by_wl, TRAIN_WORKLOADS),
                "val": expand_workloads(by_wl, VAL_WORKLOADS),
                "test": expand_workloads(by_wl, HARD_TEST_WORKLOADS),
            },
        },
        # Alias for ml/train.py which reads small_subset by default.
        "small_subset": None,
    }
    doc["small_subset"] = doc["hard_subset"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {OUT}")
    print(json.dumps(doc["hard_subset"]["workloads"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
