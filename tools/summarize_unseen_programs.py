#!/usr/bin/env python3
"""Summarize unseen-program bimodal screen + gated matrix runs."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARSED = ROOT / "results" / "parsed"
SPLIT = ROOT / "data" / "splits" / "spec_unseen_v1.json"
GATE_RE = re.compile(
    r"ANBA_GATE_STATS uncertain=(\d+) override=(\d+) override_correct=(\d+) "
    r"override_rate=([0-9.]+) override_precision=([0-9.]+)"
)
RUN_RE = re.compile(
    r"^(?P<pred>bimodal|anba_residual_hybrid|anba_transformer_hybrid)_"
    r"(?:(?P<tag>m\d+)_)?"
    r"(?P<trace>.+)\.champsimtrace\.xz_\d{8}T\d{6}Z\.json$"
)


def parse_gate(log_path: str | None) -> dict | None:
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


def load_run(path: Path) -> dict | None:
    try:
        doc = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if not doc.get("parse_ok"):
        return None
    m = doc.get("metrics") or {}
    if m.get("ipc") in (None, "unavailable"):
        return None
    return doc


def main() -> int:
    split_doc = json.loads(SPLIT.read_text()) if SPLIT.exists() else {}
    selected = set(split_doc.get("selected_traces") or [])
    screen_path = ROOT / split_doc.get("screening_results_path", "results/parsed/unseen_bimodal_screen.json")
    screen_doc = json.loads(screen_path.read_text()) if screen_path.exists() else None

    best: dict[tuple[str, str, str], tuple[str, dict]] = {}
    for path in sorted(PARSED.glob("*.json")):
        m = RUN_RE.match(path.name)
        if not m:
            continue
        trace = m.group("trace") + ".champsimtrace.xz"
        if selected and trace not in selected:
            continue
        pred = m.group("pred")
        tag = m.group("tag") or ""
        doc = load_run(path)
        if not doc:
            continue
        key = (pred, tag, trace)
        ts = path.name.rsplit("_", 1)[-1].replace(".json", "")
        prev = best.get(key)
        if prev is None or ts > prev[0]:
            best[key] = (ts, doc)

    rows: list[dict] = []
    for (pred, tag, trace), (_ts, doc) in sorted(best.items()):
        metrics = doc["metrics"]
        margin = int(tag[1:]) if tag.startswith("m") and tag[1:].isdigit() else None
        label = pred
        if pred != "bimodal" and margin is not None:
            label = f"{pred}_m{margin}"
        gate = parse_gate(doc.get("log_path"))
        rows.append(
            {
                "predictor": pred,
                "margin": margin,
                "label": label,
                "trace": trace,
                "workload": trace.split("_s-")[0],
                "warmup": doc.get("warmup_instructions"),
                "sim": doc.get("simulation_instructions"),
                "ipc": metrics.get("ipc"),
                "branch_accuracy_pct": metrics.get("branch_accuracy_pct"),
                "branch_mpki": metrics.get("branch_mpki"),
                "gate_stats": gate,
            }
        )

    base_by_trace = {r["trace"]: r for r in rows if r["predictor"] == "bimodal"}
    deltas: list[dict] = []
    for r in rows:
        base = base_by_trace.get(r["trace"])
        if not base or r["predictor"] == "bimodal":
            continue
        deltas.append(
            {
                "trace": r["trace"],
                "label": r["label"],
                "delta_ipc": round(float(r["ipc"]) - float(base["ipc"]), 4),
                "delta_branch_accuracy_pct": round(
                    float(r["branch_accuracy_pct"]) - float(base["branch_accuracy_pct"]), 2
                ),
                "override_rate": (r.get("gate_stats") or {}).get("override_rate"),
                "override_precision": (r.get("gate_stats") or {}).get("override_precision"),
            }
        )

    summary = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "split": str(SPLIT.relative_to(ROOT)),
        "selected_traces": sorted(selected),
        "bimodal_screen": screen_doc,
        "rows": rows,
        "deltas_vs_bimodal": deltas,
        "fabricated": False,
    }
    out = PARSED / "unseen_programs_matrix_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
