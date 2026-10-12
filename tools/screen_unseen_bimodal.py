#!/usr/bin/env python3
"""Bimodal-only screen on unseen candidate traces; rank weak bimodal regions."""

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
    parser.add_argument(
        "--split",
        type=Path,
        default=ROOT / "data" / "splits" / "spec_unseen_v1.json",
    )
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--sim", type=int, default=None)
    parser.add_argument("--pick", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    split_doc = json.loads(args.split.read_text())
    screen = split_doc.get("screening") or {}
    warmup = args.warmup if args.warmup is not None else int(screen.get("warmup", 50_000))
    sim = args.sim if args.sim is not None else int(screen.get("sim", 50_000))
    pick_n = args.pick if args.pick is not None else int(screen.get("pick_n", 4))
    out = args.out or ROOT / split_doc.get("screening_results_path", "results/parsed/unseen_bimodal_screen.json")

    traces = split_doc.get("candidate_traces") or []
    trace_dir = ROOT / "data" / "traces"
    bin_path = ROOT / "champsim" / "ChampSim" / "bin" / "champsim_bimodal"
    if not bin_path.exists():
        print(f"missing {bin_path}; make champsim-build PREDICTOR=bimodal", file=sys.stderr)
        return 1

    rows: list[dict] = []
    for name in traces:
        tr = trace_dir / name
        if not tr.exists():
            rows.append({"trace": name, "parse_ok": False, "error": "missing trace"})
            continue
        cmd = [
            str(bin_path),
            f"--warmup-instructions={warmup}",
            f"--simulation-instructions={sim}",
            str(tr),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            rows.append(
                {
                    "trace": name,
                    "workload": name.split("_s-")[0],
                    "parse_ok": False,
                    "error": (proc.stderr or proc.stdout)[:400],
                }
            )
            continue
        parsed = parse_log(proc.stdout)
        rows.append(
            {
                "trace": name,
                "workload": name.split("_s-")[0],
                "warmup": warmup,
                "sim": sim,
                "ipc": parsed["ipc"],
                "branch_accuracy_pct": parsed["branch_accuracy_pct"],
                "branch_mpki": parsed["branch_mpki"],
                "parse_ok": parsed["parse_ok"],
            }
        )

    ok = [r for r in rows if r.get("parse_ok")]
    ok.sort(
        key=lambda r: (
            r.get("branch_accuracy_pct") if r.get("branch_accuracy_pct") is not None else 999.0,
            -(r.get("branch_mpki") or 0.0),
        )
    )
    selected = [r["trace"] for r in ok[:pick_n]]

    doc = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "split": str(args.split.relative_to(ROOT)),
        "warmup": warmup,
        "sim": sim,
        "pick_n": pick_n,
        "rows": rows,
        "ranked_by_bimodal_weakness": [r["trace"] for r in ok],
        "selected_traces": selected,
        "fabricated": False,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n")

    split_doc["selected_traces"] = selected
    split_doc["screening"]["warmup"] = warmup
    split_doc["screening"]["sim"] = sim
    args.split.write_text(json.dumps(split_doc, indent=2) + "\n")
    print(json.dumps(doc, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
