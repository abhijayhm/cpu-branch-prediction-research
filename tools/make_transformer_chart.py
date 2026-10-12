#!/usr/bin/env python3
"""Bar chart: branch accuracy deltas vs bimodal for transformer matrix."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    summary_path = ROOT / "results" / "parsed" / "transformer_matrix_summary.json"
    if not summary_path.exists():
        print("missing transformer_matrix_summary.json", file=sys.stderr)
        return 1
    doc = json.loads(summary_path.read_text())
    deltas = doc.get("deltas_vs_bimodal") or []
    if not deltas:
        print("no deltas to plot", file=sys.stderr)
        return 1

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available", file=sys.stderr)
        return 1

    preds = ["anba_residual_hybrid", "nn_transformer_frozen", "anba_transformer_hybrid"]
    traces = sorted({d["trace"] for d in deltas})
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.22
    x = list(range(len(traces)))
    for i, pred in enumerate(preds):
        vals = []
        for tr in traces:
            match = [d for d in deltas if d["trace"] == tr and d["predictor"] == pred]
            vals.append(match[0]["delta_branch_accuracy_pct"] if match else 0.0)
        offset = (i - 1) * width
        ax.bar([xi + offset for xi in x], vals, width=width, label=pred.replace("_", " "))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([t.split(".")[0].split("_")[-1][:12] for t in traces], rotation=15, ha="right")
    ax.set_ylabel("Δ branch accuracy vs bimodal (pp)")
    ax.set_title("NN-T eval on held-out hard test traces")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = ROOT / "paper" / "figures" / "nn_t_hard_accuracy_delta.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
