#!/usr/bin/env python3
"""Run a ChampSim binary on one real trace and parse metrics."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from parse_champsim import parse_log  # noqa: E402

import json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin", required=True, type=Path)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--predictor", required=True)
    parser.add_argument("--warmup", type=int, default=100_000)
    parser.add_argument("--sim", type=int, default=200_000)
    parser.add_argument("--extra-env", action="append", default=[])
    args = parser.parse_args()

    if not args.bin.exists():
        print(f"missing binary {args.bin}", file=sys.stderr)
        return 1
    if not args.trace.exists():
        print(f"missing trace {args.trace} (download with make traces)", file=sys.stderr)
        return 1

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_dir = ROOT / "results" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_path = raw_dir / f"{args.predictor}_{args.trace.name}_{stamp}.log"

    cmd = [
        str(args.bin),
        f"--warmup-instructions={args.warmup}",
        f"--simulation-instructions={args.sim}",
        str(args.trace),
    ]
    env = dict(**{k: v for k, v in [e.split("=", 1) for e in args.extra_env]})
    print(" ".join(cmd), file=sys.stderr)
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, env={**dict(**{k: v for k, v in __import__("os").environ.items()}), **env})
    log_path.write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        print(f"ChampSim exited {proc.returncode}; log {log_path}", file=sys.stderr)
        return proc.returncode

    parsed = parse_log(proc.stdout)
    record = {
        "schema_version": 1,
        "generated_utc": stamp,
        "predictor": args.predictor,
        "trace": args.trace.name,
        "warmup_instructions": args.warmup,
        "simulation_instructions": args.sim,
        "command": cmd,
        "log_path": str(log_path),
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
    parsed_path = ROOT / "results" / "parsed" / f"{args.predictor}_{args.trace.name}_{stamp}.json"
    parsed_path.parent.mkdir(parents=True, exist_ok=True)
    parsed_path.write_text(json.dumps(record, indent=2) + "\n")
    latest = ROOT / "results" / "parsed" / f"latest_{args.predictor}.json"
    latest.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    return 0 if parsed["parse_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
