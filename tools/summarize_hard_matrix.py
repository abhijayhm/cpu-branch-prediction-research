#!/usr/bin/env python3
"""Summarize hard-eval ChampSim runs into hard_matrix_summary.json."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARSED = ROOT / "results" / "parsed"
SPLIT = ROOT / "data" / "splits" / "spec_hard_v1.json"
PREDS = ("bimodal", "instrumented", "nn_frozen", "anba_hybrid", "anba_online")
RUN_RE = re.compile(
    r"^(?P<pred>bimodal|instrumented|nn_frozen|anba_hybrid|anba_online)_"
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


def main() -> int:
    hard_traces: set[str] = set()
    if SPLIT.exists():
        split_doc = json.loads(SPLIT.read_text())
        hard_traces = set(split_doc["hard_subset"]["traces"]["test"])

    best: dict[tuple[str, str], tuple[str, dict]] = {}
    for path in sorted(PARSED.glob("*.json")):
        m = RUN_RE.match(path.name)
        if not m:
            continue
        trace = m.group("trace") + ".champsimtrace.xz"
        if hard_traces and trace not in hard_traces:
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

    # Delta table vs bimodal baseline per trace.
    bimodal_by_trace = {
        r["trace"]: r for r in rows if r["predictor"] == "bimodal"
    }
    deltas: list[dict] = []
    for r in rows:
        base = bimodal_by_trace.get(r["trace"])
        if not base or r["predictor"] == "bimodal":
            continue
        deltas.append(
            {
                "trace": r["trace"],
                "predictor": r["predictor"],
                "delta_ipc": round(float(r["ipc"]) - float(base["ipc"]), 4),
                "delta_branch_accuracy_pct": round(
                    float(r["branch_accuracy_pct"]) - float(base["branch_accuracy_pct"]), 2
                ),
                "delta_branch_mpki": round(
                    float(r["branch_mpki"]) - float(base["branch_mpki"]), 2
                ),
            }
        )

    summary = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "split": str(SPLIT.relative_to(ROOT)) if SPLIT.exists() else None,
        "hard_traces": sorted(hard_traces),
        "n_runs": len(rows),
        "predictors": list(PREDS),
        "rows": rows,
        "deltas_vs_bimodal": deltas,
        "fabricated": False,
    }
    out = PARSED / "hard_matrix_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
