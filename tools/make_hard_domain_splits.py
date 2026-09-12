#!/usr/bin/env python3
"""Freeze workload-level split for hard-domain train/val/test (spec_hard_domain_v1).

All train/val/test workloads come from the bimodal-weak pool (acc < ~90% on probe).
Trace-level split: never train and test on the same trace/workload.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "metadata" / "traces.csv"
OUT = ROOT / "data" / "splits" / "spec_hard_domain_v1.json"
SEED = 20260914

# Bimodal probe (WARMUP=1e5 SIM=2e5): acc < 90%, already downloaded for hard eval.
HARD_POOL_WORKLOADS = (
    "654.roms",
    "648.exchange2",
    "603.bwaves",
    "607.cactuBSSN",
    "631.deepsjeng",
)

# Hard-domain split: train on hard, val on held-out hard, test on held-out hard.
# ≥2 train, 1 val, ≥2 test (all disjoint workloads).
TRAIN_WORKLOADS = (
    "603.bwaves",
    "607.cactuBSSN",
)
VAL_WORKLOADS = ("631.deepsjeng",)
TEST_WORKLOADS = (
    "654.roms",
    "648.exchange2",
)

TRACE_PICK = {
    "603.bwaves": "603.bwaves_s-3699B.champsimtrace.xz",
    "607.cactuBSSN": "607.cactuBSSN_s-4248B.champsimtrace.xz",
    "631.deepsjeng": "631.deepsjeng_s-928B.champsimtrace.xz",
    "654.roms": "654.roms_s-1021B.champsimtrace.xz",
    "648.exchange2": "648.exchange2_s-1699B.champsimtrace.xz",
}

BIMODAL_PROBE = {
    "603.bwaves_s-3699B.champsimtrace.xz": 87.22,
    "607.cactuBSSN_s-4248B.champsimtrace.xz": 87.45,
    "631.deepsjeng_s-928B.champsimtrace.xz": 89.71,
    "654.roms_s-1021B.champsimtrace.xz": 80.48,
    "648.exchange2_s-1699B.champsimtrace.xz": 84.04,
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


def pick_traces(wls: tuple[str, ...]) -> list[str]:
    return [TRACE_PICK[wl] for wl in wls]


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

    all_used = TRAIN_WORKLOADS + VAL_WORKLOADS + TEST_WORKLOADS
    for wl in HARD_POOL_WORKLOADS:
        if wl not in by_wl:
            raise SystemExit(f"workload {wl} missing from catalog")
    for wl in all_used:
        if wl not in HARD_POOL_WORKLOADS:
            raise SystemExit(f"workload {wl} not in hard pool")

    train_traces = pick_traces(TRAIN_WORKLOADS)
    val_traces = pick_traces(VAL_WORKLOADS)
    test_traces = pick_traces(TEST_WORKLOADS)

    subset = {
        "workloads_considered": {
            "pool": list(HARD_POOL_WORKLOADS),
            "train": list(TRAIN_WORKLOADS),
            "val": list(VAL_WORKLOADS),
            "test": list(TEST_WORKLOADS),
        },
        "workloads": {
            "train": list(TRAIN_WORKLOADS),
            "val": list(VAL_WORKLOADS),
            "test": list(TEST_WORKLOADS),
        },
        "traces": {
            "train": train_traces,
            "val": val_traces,
            "test": test_traces,
        },
        "full_workload_traces": {
            "train": expand_workloads(by_wl, TRAIN_WORKLOADS),
            "val": expand_workloads(by_wl, VAL_WORKLOADS),
            "test": expand_workloads(by_wl, TEST_WORKLOADS),
        },
        "bimodal_probe_acc_pct": {
            trace: BIMODAL_PROBE[trace]
            for trace in train_traces + val_traces + test_traces
        },
    }

    doc = {
        "name": "spec_hard_domain_v1",
        "seed": SEED,
        "unit": "workload",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_catalog": "data/metadata/traces.csv",
        "source_doi": "10.5281/zenodo.10960004",
        "parent_split": "data/splits/spec_hard_v1.json",
        "rules": [
            "Split is workload-level (SPEC benchmark), not instruction-level.",
            "All simpoints of a workload share the same split.",
            "Train/val/test workloads are all from the bimodal-weak pool (probe acc < ~90%).",
            "Never train and test on the same trace/workload.",
            "Dataset rows are all resolved branches with true outcomes (not failure-only).",
        ],
        "dataset_note": (
            "Each instrumentation CSV row is a resolved branch with its true taken/not-taken "
            "outcome. Supervised learning already teaches the correct bit everywhere, including "
            "where bimodal and other heuristics fail."
        ),
        "bimodal_probe": {
            "warmup": 100000,
            "sim": 200000,
            "source": "results/parsed/bimodal_probe.json",
            "threshold_pct": 90.0,
        },
        "hard_domain_subset": subset,
        "small_subset": None,
    }
    doc["small_subset"] = doc["hard_domain_subset"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {OUT}")
    print(json.dumps(doc["hard_domain_subset"]["workloads"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
