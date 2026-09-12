#!/usr/bin/env python3
"""Summarize parsed ChampSim runs under results/parsed/ into one JSON table."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARSED = ROOT / "results" / "parsed"
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
    best: dict[tuple[str, str], tuple[str, dict]] = {}
    for path in sorted(PARSED.glob("*.json")):
        m = RUN_RE.match(path.name)
        if not m:
            continue
        doc = load_run(path)
        if not doc:
            continue
        key = (m.group("pred"), m.group("trace") + ".champsimtrace.xz")
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
    summary = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_runs": len(rows),
        "predictors": list(PREDS),
        "rows": rows,
        "fabricated": False,
    }
    out = PARSED / "matrix_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
