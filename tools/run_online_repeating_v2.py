#!/usr/bin/env python3
"""Persistent multi-epoch repeating workload matrix (v2)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from run_online_epochs import run_one  # noqa: E402


def main() -> int:
    split = json.loads((ROOT / "data/splits/online_repeating_v2.json").read_text())
    epochs = int(split["epochs"])
    warmup = int(split["warmup"])
    sim = int(split["sim"])
    tuned = split.get("tuned_params") or {}
    traces = [ROOT / "data/traces" / t for t in split["test_traces"]]
    preds = split["predictors"]
    ckpt_dir = ROOT / "results" / "checkpoints" / "v2"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out = ROOT / "results" / "parsed" / "online_repeating_v2_runs.jsonl"

    persist_preds = {"gshare_repeat", "anba_gshare_online", "anba_online_repeat_chooser", "anba_online_repeat_vote"}
    rows = []
    for tr in traces:
        if not tr.exists():
            print(f"missing {tr}", file=sys.stderr)
            return 1
        for pred in preds:
            ckpt = ckpt_dir / f"{pred}_{tr.stem}.bin"
            if ckpt.exists():
                ckpt.unlink()
            extra_base: dict[str, str] = {}
            if pred == "anba_gshare_online":
                extra_base = {k: str(v) for k, v in tuned.items()}
            for ep in range(1, epochs + 1):
                extra = {
                    "ANBA_EPOCH": str(ep),
                    **extra_base,
                }
                if pred in persist_preds:
                    extra["ANBA_PERSIST_STATE"] = "1"
                    extra["ANBA_CHECKPOINT_PATH"] = str(ckpt)
                bin_path = ROOT / "champsim" / "ChampSim" / "bin" / f"champsim_{pred}"
                row = run_one(bin_path, tr, pred, warmup, sim, extra)
                rows.append(row)

    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps({"n_rows": len(rows), "out": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
