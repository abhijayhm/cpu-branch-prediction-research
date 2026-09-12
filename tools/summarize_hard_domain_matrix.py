#!/usr/bin/env python3
"""Summarize hard-domain test ChampSim runs into hard_domain_matrix_summary.json."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARSED = ROOT / "results" / "parsed"
SPLIT = ROOT / "data" / "splits" / "spec_hard_domain_v1.json"
EASY_HARD_SUMMARY = PARSED / "hard_matrix_summary.json"
PREDS = ("bimodal", "nn_frozen", "anba_hybrid", "anba_online")
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


def load_train_reports() -> list[dict]:
    reports = []
    for arch in ("nn_a", "nn_b", "nn_c"):
        path = PARSED / f"train_hard_domain_{arch}.json"
        if path.exists():
            reports.append(json.loads(path.read_text()))
    return reports


def contrast_table(hard_domain_deltas: list[dict]) -> list[dict]:
    """Compare easy-train→hard-test vs hard-train→hard-test deltas per trace/predictor."""
    if not EASY_HARD_SUMMARY.exists():
        return []
    easy_doc = json.loads(EASY_HARD_SUMMARY.read_text())
    easy_by_key = {
        (d["trace"], d["predictor"]): d for d in easy_doc.get("deltas_vs_bimodal", [])
    }
    out = []
    for d in hard_domain_deltas:
        key = (d["trace"], d["predictor"])
        easy = easy_by_key.get(key)
        if not easy:
            continue
        out.append(
            {
                "trace": d["trace"],
                "predictor": d["predictor"],
                "easy_train_delta_acc_pct": easy["delta_branch_accuracy_pct"],
                "hard_domain_delta_acc_pct": d["delta_branch_accuracy_pct"],
                "acc_pct_improvement_vs_easy_train": round(
                    d["delta_branch_accuracy_pct"] - easy["delta_branch_accuracy_pct"], 2
                ),
                "easy_train_delta_ipc": easy["delta_ipc"],
                "hard_domain_delta_ipc": d["delta_ipc"],
                "ipc_improvement_vs_easy_train": round(d["delta_ipc"] - easy["delta_ipc"], 4),
            }
        )
    return out


def main() -> int:
    test_traces: set[str] = set()
    split_doc = {}
    if SPLIT.exists():
        split_doc = json.loads(SPLIT.read_text())
        test_traces = set(split_doc["hard_domain_subset"]["traces"]["test"])

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

    bimodal_by_trace = {r["trace"]: r for r in rows if r["predictor"] == "bimodal"}
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
                "delta_branch_mpki": round(float(r["branch_mpki"]) - float(base["branch_mpki"]), 2),
            }
        )

    train_reports = load_train_reports()
    summary = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "split": str(SPLIT.relative_to(ROOT)) if SPLIT.exists() else None,
        "train_workloads": split_doc.get("hard_domain_subset", {}).get("workloads", {}).get("train", []),
        "val_workloads": split_doc.get("hard_domain_subset", {}).get("workloads", {}).get("val", []),
        "test_traces": sorted(test_traces),
        "warmup": 100000,
        "sim": 200000,
        "n_runs": len(rows),
        "predictors": list(PREDS),
        "rows": rows,
        "deltas_vs_bimodal": deltas,
        "offline_train_reports": train_reports,
        "contrast_easy_train_vs_hard_domain_train": contrast_table(deltas),
        "fabricated": False,
    }
    out = PARSED / "hard_domain_matrix_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
