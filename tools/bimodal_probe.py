#!/usr/bin/env python3
"""Run a short bimodal probe on downloaded traces and rank by branch accuracy."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from parse_champsim import parse_log  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=100_000)
    parser.add_argument("--sim", type=int, default=200_000)
    parser.add_argument("--trace-dir", type=Path, default=ROOT / "data" / "traces")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "parsed" / "bimodal_probe.json")
    args = parser.parse_args()

    bin_path = ROOT / "champsim" / "ChampSim" / "bin" / "champsim_bimodal"
    if not bin_path.exists():
        print(f"missing {bin_path}; build with: make champsim-build PREDICTOR=bimodal", file=sys.stderr)
        return 1

    traces = sorted(args.trace_dir.glob("*.champsimtrace.xz"))
    if not traces:
        print(f"no traces in {args.trace_dir}", file=sys.stderr)
        return 1

    rows = []
    for tr in traces:
        cmd = [
            str(bin_path),
            f"--warmup-instructions={args.warmup}",
            f"--simulation-instructions={args.sim}",
            str(tr),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            rows.append({"trace": tr.name, "error": proc.stderr[:500], "parse_ok": False})
            continue
        parsed = parse_log(proc.stdout)
        rows.append(
            {
                "trace": tr.name,
                "workload": tr.name.split("_s-")[0] if "_s-" in tr.name else tr.stem,
                "warmup": args.warmup,
                "sim": args.sim,
                "ipc": parsed["ipc"],
                "branch_accuracy_pct": parsed["branch_accuracy_pct"],
                "branch_mpki": parsed["branch_mpki"],
                "parse_ok": parsed["parse_ok"],
            }
        )

    ok = [r for r in rows if r.get("parse_ok")]
    ok.sort(key=lambda r: (r.get("branch_accuracy_pct") or 999))

    doc = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "warmup": args.warmup,
        "sim": args.sim,
        "n_traces": len(rows),
        "rows": rows,
        "ranked_by_bimodal_acc_asc": [r["trace"] for r in ok],
        "fabricated": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2) + "\n")
    print(json.dumps(doc, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
