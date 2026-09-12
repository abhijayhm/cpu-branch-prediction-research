#!/usr/bin/env python3
"""Verify PC / history / outcome instrumentation.

Checks, on a real dump:
  1. rows exist
  2. taken is 0/1
  3. history is a 16-bit GHR that updates as (h<<1|taken) & mask
  4. PCs are not all zero
Optional: compare an instrumentation CSV against a standalone trace extract.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HIST_MASK = (1 << 16) - 1


def load_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def verify(rows: list[dict]) -> dict:
    notes = []
    ok = True
    if not rows:
        return {"ok": False, "notes": ["no rows"], "n": 0}

    pcs = []
    mismatches = 0
    bad_taken = 0
    ghr_ok = 0
    expected = None
    for row in rows:
        pc = int(row["pc"], 0)
        hist = int(row["history"], 0)
        taken = int(row["taken"], 0)
        pcs.append(pc)
        if taken not in (0, 1):
            bad_taken += 1
        if expected is not None:
            if hist == expected:
                ghr_ok += 1
            else:
                mismatches += 1
        expected = ((hist << 1) | taken) & HIST_MASK

    nonzero_pc = sum(1 for p in pcs if p != 0)
    unique_pc = len(set(pcs))
    if bad_taken:
        ok = False
        notes.append(f"non-binary taken values: {bad_taken}")
    if mismatches:
        ok = False
        notes.append(f"GHR update mismatches: {mismatches}")
    else:
        notes.append("GHR update rule held for all consecutive rows")
    if nonzero_pc == 0:
        ok = False
        notes.append("all PCs were zero")
    if unique_pc < 2:
        notes.append("fewer than 2 unique PCs")

    return {
        "ok": ok,
        "n": len(rows),
        "unique_pc": unique_pc,
        "nonzero_pc": nonzero_pc,
        "ghr_matches": ghr_ok,
        "ghr_mismatches": mismatches,
        "taken_frac": sum(int(r["taken"]) for r in rows) / len(rows),
        "notes": notes,
    }


def compare(a: list[dict], b: list[dict], n: int) -> dict:
    n = min(n, len(a), len(b))
    match = 0
    for i in range(n):
        if int(a[i]["pc"], 0) == int(b[i]["pc"], 0) and int(a[i]["taken"], 0) == int(b[i]["taken"], 0):
            match += 1
    return {"compared": n, "pc_taken_matches": match, "match_frac": (match / n) if n else None}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "parsed" / "instrumentation_verify.json")
    args = parser.parse_args()
    if not args.dump.exists():
        print(f"missing {args.dump}", file=sys.stderr)
        return 1
    rows = load_rows(args.dump)
    report = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dump": str(args.dump),
        "verification": verify(rows),
        "fabricated": False,
    }
    if args.reference and args.reference.exists():
        ref = load_rows(args.reference)
        report["reference_compare"] = compare(rows, ref, 10000)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["verification"]["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
