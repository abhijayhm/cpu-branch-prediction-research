#!/usr/bin/env python3
"""Extract (PC, history, outcome) from ChampSim traces or instrumentation CSVs.

Trace layout is the official ChampSim input_instr from the pinned tree
(inc/trace_instruction.h): 64-byte records. History is reconstructed with a
16-bit global shift register — the same update as predictor/instrumented.
"""

from __future__ import annotations

import argparse
import csv
import json
import lzma
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD = struct.Struct("<QBB2B4B2Q4Q")  # 8+1+1+2+4+16+32 = 64
HIST_BITS = 16
HIST_MASK = (1 << HIST_BITS) - 1


def iter_trace(path: Path):
    opener = lzma.open if path.suffix == ".xz" or path.name.endswith(".xz") else open
    with opener(path, "rb") as f:
        while True:
            buf = f.read(RECORD.size)
            if not buf:
                break
            if len(buf) != RECORD.size:
                break
            yield RECORD.unpack(buf)


def extract_trace(path: Path, cap: int | None) -> list[tuple[int, int, int, int]]:
    rows = []
    ghist = 0
    for rec in iter_trace(path):
        ip = rec[0]
        is_branch = rec[1]
        taken = rec[2]
        if not is_branch:
            continue
        dest_regs = rec[3:5]
        src_regs = rec[5:9]
        # ChampSim classifies conditional vs always-taken using registers.
        # Keep all marked branches; branch_type is left 255 if unknown here.
        rows.append((ip, ghist, 1 if taken else 0, 255))
        ghist = ((ghist << 1) | (1 if taken else 0)) & HIST_MASK
        if cap is not None and len(rows) >= cap:
            break
        _ = dest_regs, src_regs
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--cap", type=int, default=500_000)
    args = parser.parse_args()
    if not args.trace.exists():
        print(f"missing {args.trace}", file=sys.stderr)
        return 1
    rows = extract_trace(args.trace, args.cap)
    out = args.out
    if out is None:
        stem = args.trace.name.replace(".champsimtrace.xz", "").replace(".champsimtrace", "")
        out = ROOT / "data" / "extracted" / f"{stem}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pc", "history", "taken", "branch_type"])
        w.writerows(rows)
    taken = sum(r[2] for r in rows)
    meta = {
        "trace": args.trace.name,
        "n_branches": len(rows),
        "taken": taken,
        "taken_frac": (taken / len(rows)) if rows else None,
        "out": str(out),
        "record_bytes": RECORD.size,
        "fabricated": False,
    }
    print(json.dumps(meta, indent=2))
    return 0 if rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
