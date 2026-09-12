#!/usr/bin/env python3
"""Parse ChampSim stdout for IPC, cycles, branch MPKI, and accuracy.

Writes only values present in the log. Missing fields are marked unavailable.
Never fabricates numbers.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ChampSim  (pinned commit) prints lines like:
#   CPU 0 cumulative IPC: 1.234 instructions: 1000000 cycles: 810373
#   CPU 0 Branch Prediction Accuracy: 96.12% MPKI: 4.23 Average ROB Occupancy at Mispredict: 80.12
IPC_RE = re.compile(
    r"CPU\s+(\d+)\s+cumulative IPC:\s+([0-9.]+)\s+instructions:\s+(\d+)\s+cycles:\s+(\d+)"
)
BR_RE = re.compile(
    r"CPU\s+(\d+)\s+Branch Prediction Accuracy:\s+([0-9.]+)%\s+MPKI:\s+([0-9.]+)"
)


def parse_log(text: str) -> dict:
    ipcs = list(IPC_RE.finditer(text))
    brs = list(BR_RE.finditer(text))
    result: dict = {
        "ipc": None,
        "instructions": None,
        "cycles": None,
        "branch_accuracy_pct": None,
        "branch_mpki": None,
        "parse_ok": False,
        "notes": [],
    }
    if ipcs:
        m = ipcs[-1]
        result["ipc"] = float(m.group(2))
        result["instructions"] = int(m.group(3))
        result["cycles"] = int(m.group(4))
    else:
        result["notes"].append("cumulative IPC line not found")
    if brs:
        m = brs[-1]
        result["branch_accuracy_pct"] = float(m.group(2))
        result["branch_mpki"] = float(m.group(3))
    else:
        result["notes"].append("branch prediction accuracy line not found")
    result["parse_ok"] = result["ipc"] is not None and result["branch_accuracy_pct"] is not None
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--predictor", default="unknown")
    parser.add_argument("--trace", default="unknown")
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--sim", type=int, default=None)
    args = parser.parse_args()

    text = args.log.read_text(errors="replace")
    parsed = parse_log(text)
    record = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "predictor": args.predictor,
        "trace": args.trace,
        "warmup_instructions": args.warmup,
        "simulation_instructions": args.sim,
        "log_path": str(args.log),
        "metrics": {
            "ipc": parsed["ipc"] if parsed["ipc"] is not None else "unavailable",
            "instructions": parsed["instructions"] if parsed["instructions"] is not None else "unavailable",
            "cycles": parsed["cycles"] if parsed["cycles"] is not None else "unavailable",
            "branch_accuracy_pct": parsed["branch_accuracy_pct"]
            if parsed["branch_accuracy_pct"] is not None
            else "unavailable",
            "branch_mpki": parsed["branch_mpki"] if parsed["branch_mpki"] is not None else "unavailable",
        },
        "parse_ok": parsed["parse_ok"],
        "notes": parsed["notes"],
        "fabricated": False,
    }
    out = args.out
    if out is None:
        stem = args.log.stem
        out = ROOT / "results" / "parsed" / f"{stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    return 0 if parsed["parse_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
