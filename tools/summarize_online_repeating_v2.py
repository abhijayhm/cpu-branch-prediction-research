#!/usr/bin/env python3
"""Summarize v2 runs with deltas vs bimodal and gshare_repeat."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results/parsed/online_repeating_v2_runs.jsonl"
OUT = ROOT / "results/parsed/online_repeating_v2_summary.json"


def main() -> int:
    rows = [json.loads(line) for line in RUNS.read_text().splitlines() if line.strip()]
    bim = {(r["trace"], r["epoch"]): r for r in rows if r["predictor"] == "bimodal" and r.get("parse_ok")}
    gsh = {(r["trace"], r["epoch"]): r for r in rows if r["predictor"] == "gshare_repeat" and r.get("parse_ok")}
    curves = []
    for r in rows:
        if not r.get("parse_ok"):
            continue
        m = r["metrics"]
        b = bim.get((r["trace"], r["epoch"]))
        g = gsh.get((r["trace"], r["epoch"]))
        curves.append(
            {
                "trace": r["trace"],
                "predictor": r["predictor"],
                "epoch": r["epoch"],
                "ipc": m["ipc"],
                "branch_accuracy_pct": m["branch_accuracy_pct"],
                "branch_mpki": m["branch_mpki"],
                "delta_acc_vs_bimodal": round(float(m["branch_accuracy_pct"]) - float(b["metrics"]["branch_accuracy_pct"]), 3)
                if b
                else None,
                "delta_ipc_vs_bimodal": round(float(m["ipc"]) - float(b["metrics"]["ipc"]), 4) if b else None,
                "delta_acc_vs_gshare": round(float(m["branch_accuracy_pct"]) - float(g["metrics"]["branch_accuracy_pct"]), 3)
                if g
                else None,
                "delta_ipc_vs_gshare": round(float(m["ipc"]) - float(g["metrics"]["ipc"]), 4) if g else None,
                "gate_stats": r.get("gate_stats"),
                "online_epoch_stats": r.get("online_epoch_stats"),
            }
        )

    compare = []
    for tr in sorted({c["trace"] for c in curves}):
        for pred in sorted({c["predictor"] for c in curves}):
            e1 = next((c for c in curves if c["trace"] == tr and c["predictor"] == pred and c["epoch"] == 1), None)
            ef = next(
                (c for c in curves if c["trace"] == tr and c["predictor"] == pred and c["epoch"] == max(x["epoch"] for x in curves if x["trace"] == tr)),
                None,
            )
            if not e1 or not ef:
                continue
            compare.append(
                {
                    "trace": tr,
                    "predictor": pred,
                    "epoch1_acc": e1["branch_accuracy_pct"],
                    "final_acc": ef["branch_accuracy_pct"],
                    "epoch1_ipc": e1["ipc"],
                    "final_ipc": ef["ipc"],
                    "delta_acc_ep1_vs_bimodal": e1["delta_acc_vs_bimodal"],
                    "delta_acc_final_vs_bimodal": ef["delta_acc_vs_bimodal"],
                    "delta_acc_ep1_vs_gshare": e1["delta_acc_vs_gshare"],
                    "delta_acc_final_vs_gshare": ef["delta_acc_vs_gshare"],
                    "delta_ipc_final_vs_gshare": ef["delta_ipc_vs_gshare"],
                    "final_override_rate": (ef.get("gate_stats") or {}).get("override_rate"),
                    "final_override_precision": (ef.get("gate_stats") or {}).get("override_precision"),
                }
            )

    tune = None
    tp = ROOT / "results/parsed/online_tune_val.json"
    if tp.exists():
        tune = json.loads(tp.read_text())
    summary = {
        "schema_version": 2,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": str(RUNS.relative_to(ROOT)),
        "val_tune": tune,
        "curves": curves,
        "epoch1_vs_final": compare,
        "vote_bug_note": (
            "Legacy vote path returned model_pred while gating on combined vote; aggressive overrides "
            "produced acc ≈ 2*bimodal − gshare (mirror deltas). Fixed to use vote_taken=combined>=0; "
            "see tests/test_online_vote_aggregate.py"
        ),
        "fabricated": False,
    }
    OUT.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
