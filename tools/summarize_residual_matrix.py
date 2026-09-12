#!/usr/bin/env python3
"""Summarize residual-matrix ChampSim runs into residual_matrix_summary.json."""

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
    "anba_hybrid",
    "anba_online",
    "anba_residual_hybrid",
    "nn_frozen",
)
RUN_RE = re.compile(
    r"^(?P<pred>bimodal|instrumented|nn_frozen|anba_hybrid|anba_online|anba_residual_hybrid)_"
    r"(?P<trace>.+)\.champsimtrace\.xz_\d{8}T\d{6}Z\.json$"
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


def load_train_summary() -> dict | None:
    path = PARSED / "train_residual_summary.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def load_dataset_stats() -> dict | None:
    path = PARSED / "residual_dataset_stats.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


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
    split_doc = {}
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
        if m.group("pred") not in PREDS:
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
        rows.append(
            {
                "predictor": pred,
                "trace": trace,
                "warmup": doc.get("warmup_instructions"),
                "sim": doc.get("simulation_instructions"),
                "ipc": metrics.get("ipc"),
                "branch_accuracy_pct": metrics.get("branch_accuracy_pct"),
                "branch_mpki": metrics.get("branch_mpki"),
                "log_path": doc.get("log_path"),
            }
        )

    deltas_vs_bimodal = delta_table(rows, "bimodal")
    deltas_vs_hybrid = delta_table(rows, "anba_hybrid")

    summary = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "split": str(SPLIT.relative_to(ROOT)) if SPLIT.exists() else None,
        "test_traces": sorted(test_traces),
        "warmup": 100000,
        "sim": 200000,
        "nn_margin_default": 8,
        "residual_model": "models/export/nn_residual_int8.bin",
        "n_runs": len(rows),
        "predictors": list(PREDS),
        "rows": rows,
        "deltas_vs_bimodal": deltas_vs_bimodal,
        "deltas_vs_anba_hybrid": deltas_vs_hybrid,
        "offline_train_summary": load_train_summary(),
        "residual_dataset_stats": load_dataset_stats(),
        "fabricated": False,
    }
    out = PARSED / "residual_matrix_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
