#!/usr/bin/env python3
"""Charts for online repeating v2 vs bimodal and gshare."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results/parsed/online_repeating_v2_summary.json"


def main() -> int:
    import matplotlib.pyplot as plt

    doc = json.loads(SUMMARY.read_text())
    curves = doc["curves"]
    focus = ("anba_gshare_online", "gshare_repeat", "hashed_perceptron", "anba_online_repeat_vote")
    traces = sorted({c["trace"] for c in curves})
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
    for ax, tr in zip(axes.flatten(), traces):
        for pred in focus:
            pts = sorted(
                [(c["epoch"], c["delta_acc_vs_gshare"]) for c in curves if c["trace"] == tr and c["predictor"] == pred],
                key=lambda x: x[0],
            )
            if not pts or pts[0][1] is None:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", label=pred.replace("anba_gshare_online", "hybrid").replace("anba_online_repeat_", "ol_"))
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_title(tr.split("_s-")[0])
        ax.set_xlabel("epoch")
        ax.set_ylabel("Δ acc vs gshare (pp)")
    for j in range(len(traces), 6):
        axes.flatten()[j].axis("off")
    fig.suptitle("Persistent repeating tasks (v2)")
    fig.legend(*axes.flatten()[0].get_legend_handles_labels(), loc="upper center", ncol=4, fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = ROOT / "paper/figures/online_repeating_v2_delta_gshare.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")

    fig2, axes2 = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
    for ax, tr in zip(axes2.flatten(), traces):
        for pred in ("anba_gshare_online", "gshare_repeat"):
            pts = sorted(
                [(c["epoch"], c["delta_acc_vs_bimodal"]) for c in curves if c["trace"] == tr and c["predictor"] == pred],
                key=lambda x: x[0],
            )
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", label=pred)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_title(tr.split("_s-")[0])
    fig2.suptitle("Δ acc vs bimodal")
    fig2.tight_layout()
    out2 = ROOT / "paper/figures/online_repeating_v2_delta_bimodal.png"
    fig2.savefig(out2, dpi=140)
    print(f"wrote {out2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
