#!/usr/bin/env python3
"""Run K replay epochs; chain online checkpoints across epochs."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from parse_champsim import parse_log  # noqa: E402

GATE_RE = re.compile(
    r"ANBA_GATE_STATS uncertain=(\d+) override=(\d+) override_correct=(\d+) "
    r"override_rate=([0-9.]+) override_precision=([0-9.]+)"
)
EPOCH_RE = re.compile(
    r"ANBA_ONLINE_EPOCH_STATS epoch=(\d+) train_updates=(\d+) uncertain=(\d+) override=(\d+) "
    r"override_correct=(\d+) override_rate=([0-9.]+) override_precision=([0-9.]+)"
)


def parse_telemetry(text: str) -> dict:
    out: dict = {}
    m = GATE_RE.search(text)
    if m:
        out["gate_stats"] = {
            "uncertain_branches": int(m.group(1)),
            "overrides": int(m.group(2)),
            "override_correct": int(m.group(3)),
            "override_rate": float(m.group(4)),
            "override_precision": float(m.group(5)),
        }
    m = EPOCH_RE.search(text)
    if m:
        out["online_epoch_stats"] = {
            "epoch": int(m.group(1)),
            "train_updates": int(m.group(2)),
            "uncertain_branches": int(m.group(3)),
            "overrides": int(m.group(4)),
            "override_correct": int(m.group(5)),
            "override_rate": float(m.group(6)),
            "override_precision": float(m.group(7)),
        }
    return out


def run_one(
    bin_path: Path,
    trace: Path,
    predictor: str,
    warmup: int,
    sim: int,
    extra_env: dict[str, str],
) -> dict:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_dir = ROOT / "results" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_path = raw_dir / f"{predictor}_{trace.name}_ep{extra_env.get('ANBA_EPOCH', '0')}_{stamp}.log"
    cmd = [
        str(bin_path),
        f"--warmup-instructions={warmup}",
        f"--simulation-instructions={sim}",
        str(trace),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env={**dict(__import__("os").environ), **extra_env},
    )
    log_path.write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"{predictor} failed: {proc.stderr[:500]}")
    parsed = parse_log(proc.stdout)
    telem = parse_telemetry(proc.stdout + proc.stderr)
    return {
        "predictor": predictor,
        "trace": trace.name,
        "epoch": int(extra_env.get("ANBA_EPOCH", "1")),
        "warmup": warmup,
        "sim": sim,
        "log_path": str(log_path),
        "metrics": {
            "ipc": parsed["ipc"],
            "branch_accuracy_pct": parsed["branch_accuracy_pct"],
            "branch_mpki": parsed["branch_mpki"],
        },
        "parse_ok": parsed["parse_ok"],
        **telem,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, default=ROOT / "data" / "splits" / "online_repeating_v1.json")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "parsed" / "online_repeating_runs.jsonl")
    args = parser.parse_args()
    doc = json.loads(args.split.read_text())
    epochs = int(doc.get("epochs", 5))
    warmup = int(doc.get("warmup", 100_000))
    sim = int(doc.get("sim", 200_000))
    traces = [ROOT / "data" / "traces" / t for t in doc["traces"]]
    residual = ROOT / doc.get("residual_model", "models/export/nn_residual_int8.bin")
    margin = str(doc.get("residual_margin", 20))
    ckpt_dir = ROOT / "results" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    static_preds = doc["predictors"]["static_per_epoch"]
    online_preds = doc["predictors"]["online_chained"]

    for tr in traces:
        if not tr.exists():
            print(f"missing {tr}", file=sys.stderr)
            return 1
        for pred in static_preds:
            bin_path = ROOT / "champsim" / "ChampSim" / "bin" / f"champsim_{pred}"
            extra: dict[str, str] = {}
            if pred == "anba_residual_hybrid":
                extra["ANBA_MODEL_PATH"] = str(residual)
                extra["ANBA_NN_MARGIN"] = margin
            for ep in range(1, epochs + 1):
                extra["ANBA_EPOCH"] = str(ep)
                row = run_one(bin_path, tr, pred, warmup, sim, extra)
                rows.append(row)

        for pred in online_preds:
            bin_path = ROOT / "champsim" / "ChampSim" / "bin" / f"champsim_{pred}"
            ckpt = ckpt_dir / f"{pred}_{tr.stem}.bin"
            if ckpt.exists():
                ckpt.unlink()
            for ep in range(1, epochs + 1):
                extra = {
                    "ANBA_EPOCH": str(ep),
                    "ANBA_CHECKPOINT_PATH": str(ckpt),
                }
                row = run_one(bin_path, tr, pred, warmup, sim, extra)
                rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps({"n_rows": len(rows), "out": str(args.out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
