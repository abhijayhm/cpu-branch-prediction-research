#!/usr/bin/env python3
"""Summarize transformer eval matrix + gate stats from sim logs."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARSED = ROOT / "results" / "parsed"
SPLIT = ROOT / "data" / "splits" / "spec_residual_v1.json"
PREDS = (
    "bimodal",
    "anba_residual_hybrid",
    "nn_transformer_frozen",
    "anba_transformer_hybrid",
)
MATRIX_PREDS = (
    "bimodal",
    "anba_residual_hybrid",
    "anba_transformer_hybrid",
)
RUN_RE = re.compile(
    r"^(?P<pred>bimodal|anba_residual_hybrid|nn_transformer_frozen|anba_transformer_hybrid)_"
    r"(?P<trace>.+)\.champsimtrace\.xz_\d{8}T\d{6}Z\.json$"
)
GATE_RE = re.compile(
    r"ANBA_GATE_STATS uncertain=(\d+) override=(\d+) override_correct=(\d+) "
    r"override_rate=([0-9.]+) override_precision=([0-9.]+)"
)


def load_run(path: Path) -> dict | None:
    try:
        doc = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if not doc.get("parse_ok"):
        return None
    metrics = doc.get("metrics") or {}
    if metrics.get("ipc") in (None, "unavailable"):
        return None
    return doc


def parse_gate_stats(log_path: str | None) -> dict | None:
    if not log_path:
        return None
    p = Path(log_path)
    if not p.exists():
        return None
    text = p.read_text(errors="replace")
    m = GATE_RE.search(text)
    if not m:
        return None
    return {
        "uncertain_branches": int(m.group(1)),
        "overrides": int(m.group(2)),
        "override_correct": int(m.group(3)),
        "override_rate": float(m.group(4)),
        "override_precision": float(m.group(5)),
    }


def delta_table(rows: list[dict], baseline_pred: str) -> list[dict]:
    base_by_trace = {r["trace"]: r for r in rows if r["predictor"] == baseline_pred}
    deltas: list[dict] = []
    for r in rows:
        base = base_by_trace.get(r["trace"])
        if not base or r["predictor"] == baseline_pred:
            continue
        deltas.append(
            {
                "trace": r["trace"],
                "predictor": r["predictor"],
                "baseline": baseline_pred,
                "delta_ipc": round(float(r["ipc"]) - float(base["ipc"]), 4),
                "delta_branch_accuracy_pct": round(
                    float(r["branch_accuracy_pct"]) - float(base["branch_accuracy_pct"]), 2
                ),
                "delta_branch_mpki": round(float(r["branch_mpki"]) - float(base["branch_mpki"]), 2),
            }
        )
    return deltas


def main() -> int:
    test_traces: set[str] = set()
    if SPLIT.exists():
        split_doc = json.loads(SPLIT.read_text())
        sub = split_doc.get("residual_subset") or split_doc.get("hard_domain_subset", {})
        test_traces = set(sub.get("traces", {}).get("test", []))

    best: dict[tuple[str, str], tuple[str, dict]] = {}
    for path in sorted(PARSED.glob("*.json")):
        m = RUN_RE.match(path.name)
        if not m:
            continue
        trace = m.group("trace") + ".champsimtrace.xz"
        if test_traces and trace not in test_traces:
            continue
        doc = load_run(path)
        if not doc:
            continue
        key = (m.group("pred"), trace)
        ts = path.name.rsplit("_", 1)[-1].replace(".json", "")
        prev = best.get(key)
        if prev is None or ts > prev[0]:
            best[key] = (ts, doc)

    rows: list[dict] = []
    for (pred, trace), (_ts, doc) in sorted(best.items()):
        metrics = doc["metrics"]
        gate = parse_gate_stats(doc.get("log_path"))
        rows.append(
            {
                "predictor": pred,
                "trace": trace,
                "warmup": doc.get("warmup_instructions"),
                "sim": doc.get("simulation_instructions"),
                "ipc": metrics.get("ipc"),
                "branch_accuracy_pct": metrics.get("branch_accuracy_pct"),
                "branch_mpki": metrics.get("branch_mpki"),
                "gate_stats": gate,
                "log_path": doc.get("log_path"),
            }
        )

    train_nn_t = None
    p = PARSED / "train_residual_nn_t.json"
    if p.exists():
        train_nn_t = json.loads(p.read_text())

    summary = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "split": str(SPLIT.relative_to(ROOT)) if SPLIT.exists() else None,
        "test_traces": sorted(test_traces),
        "timing_notes": {
            "anba_transformer_hybrid_100k_200k_roms_sec": 12,
            "nn_transformer_frozen_100k_200k_roms_sec": ">240 (aborted)",
            "nn_transformer_frozen_10k_20k_roms_sec": 34,
            "full_matrix_frozen_skipped": "Use experiments/transformer_matrix_smoke.sh for frozen-only reduced window.",
        },
        "predictors": list(MATRIX_PREDS),
        "rows": [r for r in rows if r["predictor"] in MATRIX_PREDS],
        "all_rows": rows,
        "deltas_vs_bimodal": delta_table(rows, "bimodal"),
        "offline_train_nn_t": train_nn_t,
        "fabricated": False,
    }
    out = PARSED / "transformer_matrix_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
