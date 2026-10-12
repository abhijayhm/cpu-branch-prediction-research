#!/usr/bin/env python3
"""Learning-curve chart: delta acc vs bimodal per epoch."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "parsed" / "online_repeating_summary.json"
OUT = ROOT / "paper" / "figures" / "online_repeating_accuracy_delta.png"

FOCUS = (
    "anba_online_repeat_chooser",
    "anba_online_repeat_vote",
    "anba_residual_hybrid",
    "gshare",
)


def main() -> int:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available", file=__import__("sys").stderr)
        return 1
    doc = json.loads(SUMMARY.read_text())
    curves = [c for c in doc["curves"] if c["predictor"] in FOCUS]
    traces = sorted({c["trace"] for c in curves})
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
    axes_flat = axes.flatten()
    for ax, tr in zip(axes_flat, traces):
        for pred in FOCUS:
            pts = sorted(
                [(c["epoch"], c["delta_acc_vs_bimodal_epoch"]) for c in curves if c["trace"] == tr and c["predictor"] == pred],
                key=lambda x: x[0],
            )
            if not pts or pts[0][1] is None:
                continue
            xs, ys = zip(*pts)
            label = pred.replace("anba_online_repeat_", "ol_").replace("anba_residual_hybrid", "res_m20")
            ax.plot(xs, ys, marker="o", label=label)
        short = tr.split("_s-")[0]
        ax.set_title(short)
        ax.axhline(0.0, color="gray", linewidth=0.8)
        ax.set_xlabel("epoch")
        ax.set_ylabel("Δ acc vs bimodal (pp)")
    for j in range(len(traces), len(axes_flat)):
        axes_flat[j].axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8)
    fig.suptitle("Online repeating tasks (100k/200k × 5 epochs)")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
