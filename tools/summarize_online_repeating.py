#!/usr/bin/env python3
"""Summarize online repeating-task JSONL into summary + learning curves."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "parsed" / "online_repeating_runs.jsonl"
OUT = ROOT / "results" / "parsed" / "online_repeating_summary.json"


def main() -> int:
    if not RUNS.exists():
        print(f"missing {RUNS}", file=sys.stderr)
        return 1
    rows = [json.loads(line) for line in RUNS.read_text().splitlines() if line.strip()]
    bimodal = {
        (r["trace"], r["epoch"]): r
        for r in rows
        if r.get("predictor") == "bimodal" and r.get("parse_ok")
    }
    curves: list[dict] = []
    for r in rows:
        if not r.get("parse_ok"):
            continue
        base = bimodal.get((r["trace"], r["epoch"]))
        delta_acc = None
        delta_ipc = None
        if base and r["metrics"].get("branch_accuracy_pct") is not None:
            delta_acc = round(
                float(r["metrics"]["branch_accuracy_pct"]) - float(base["metrics"]["branch_accuracy_pct"]), 3
            )
            delta_ipc = round(float(r["metrics"]["ipc"]) - float(base["metrics"]["ipc"]), 4)
        curves.append(
            {
                "trace": r["trace"],
                "predictor": r["predictor"],
                "epoch": r["epoch"],
                "ipc": r["metrics"]["ipc"],
                "branch_accuracy_pct": r["metrics"]["branch_accuracy_pct"],
                "branch_mpki": r["metrics"]["branch_mpki"],
                "delta_acc_vs_bimodal_epoch": delta_acc,
                "delta_ipc_vs_bimodal_epoch": delta_ipc,
                "gate_stats": r.get("gate_stats"),
                "online_epoch_stats": r.get("online_epoch_stats"),
            }
        )

    def epoch_row(trace: str, pred: str, ep: int) -> dict | None:
        for c in curves:
            if c["trace"] == trace and c["predictor"] == pred and c["epoch"] == ep:
                return c
        return None

    traces = sorted({c["trace"] for c in curves})
    preds = sorted({c["predictor"] for c in curves})
    compare: list[dict] = []
    for tr in traces:
        for pred in preds:
            e1 = epoch_row(tr, pred, 1)
            ef = epoch_row(tr, pred, max(c["epoch"] for c in curves if c["trace"] == tr))
            if not e1 or not ef:
                continue
            compare.append(
                {
                    "trace": tr,
                    "predictor": pred,
                    "epoch1_acc": e1["branch_accuracy_pct"],
                    "final_epoch": ef["epoch"],
                    "final_acc": ef["branch_accuracy_pct"],
                    "delta_acc_epoch1_vs_bimodal": e1["delta_acc_vs_bimodal_epoch"],
                    "delta_acc_final_vs_bimodal": ef["delta_acc_vs_bimodal_epoch"],
                    "final_train_updates": (ef.get("online_epoch_stats") or {}).get("train_updates"),
                    "final_override_rate": (ef.get("gate_stats") or {}).get("override_rate"),
                    "final_override_precision": (ef.get("gate_stats") or {}).get("override_precision"),
                }
            )

    summary = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": str(RUNS.relative_to(ROOT)),
        "curves": curves,
        "epoch1_vs_final": compare,
        "fabricated": False,
    }
    OUT.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main())
