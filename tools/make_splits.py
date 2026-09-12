#!/usr/bin/env python3
"""Freeze TRACE / workload-level train/val/test splits.

Unit of split: SPEC CPU2017 workload (e.g. 648.exchange2), never individual
instructions. All simpoints of a workload stay in the same split.

Seed is fixed at 20260912. Existing data/splits/spec_v1.json is not rewritten
unless --force is given.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "metadata" / "traces.csv"
OUT = ROOT / "data" / "splits" / "spec_v1.json"
SEED = 20260912

# Small-subset workloads we actually fetch for this prototype.
SMALL_WORKLOADS = ("648.exchange2", "649.fotonik3d", "654.roms")


def workload_key(filename: str) -> str:
    if "_s-" in filename:
        return filename.split("_s-")[0]
    return filename.split(".")[0]


def assign(items: list[str], seed: int) -> dict[str, list[str]]:
    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)
    n = len(items)
    if n == 0:
        return {"train": [], "val": [], "test": []}
    if n == 1:
        return {"train": items, "val": [], "test": []}
    if n == 2:
        return {"train": [items[0]], "val": [items[1]], "test": []}
    if n == 3:
        return {"train": [items[0]], "val": [items[1]], "test": [items[2]]}
    n_train = max(1, int(round(n * 0.6)))
    n_val = max(1, int(round(n * 0.2)))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    n_test = n - n_train - n_val
    return {
        "train": items[:n_train],
        "val": items[n_train : n_train + n_val],
        "test": items[n_train + n_val : n_train + n_val + n_test],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if OUT.exists() and not args.force:
        print(f"{OUT} already frozen; use --force to regenerate")
        return 0

    if not META.exists():
        raise SystemExit(f"missing {META}; run tools/download_traces.py --catalog-only first")

    with META.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("traces.csv is empty")

    by_wl: dict[str, list[str]] = {}
    for row in rows:
        wl = row.get("workload") or workload_key(row["filename"])
        by_wl.setdefault(wl, []).append(row["filename"])

    all_workloads = sorted(by_wl)
    full = assign(all_workloads, args.seed)

    present_small = [w for w in SMALL_WORKLOADS if w in by_wl]
    # Assign small subset independently with the same seed so the prototype
    # split is stable even if the full catalog changes later.
    small = assign(sorted(present_small), args.seed)

    def expand(split: dict[str, list[str]]) -> dict:
        traces = {k: [] for k in ("train", "val", "test")}
        for part, wls in split.items():
            for wl in wls:
                traces[part].extend(sorted(by_wl[wl]))
        return {"workloads": split, "traces": traces}

    doc = {
        "name": "spec_v1",
        "seed": args.seed,
        "unit": "workload",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_catalog": "data/metadata/traces.csv",
        "source_doi": "10.5281/zenodo.10960004",
        "rules": [
            "Split is workload-level (SPEC benchmark), not instruction-level.",
            "All simpoints of a workload share the same split.",
            "Never train on the test split.",
            "Validation may be used for model selection; test is frozen.",
        ],
        "full_catalog": expand(full),
        "small_subset": {
            "workloads_considered": list(SMALL_WORKLOADS),
            **expand(small),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {OUT}")
    print(json.dumps({k: v["workloads"] for k, v in (("full", doc["full_catalog"]), ("small", doc["small_subset"]))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
