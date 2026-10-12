#!/usr/bin/env python3
"""Tune anba_gshare_online on val trace (epochs 1-2 only); pick best vs gshare_repeat."""

from __future__ import annotations

import itertools
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from run_online_epochs import parse_telemetry  # noqa: E402
from parse_champsim import parse_log  # noqa: E402


def run_epoch(
    pred: str, trace: Path, warmup: int, sim: int, epoch: int, ckpt: Path, extra: dict[str, str]
) -> float | None:
    bin_path = ROOT / "champsim" / "ChampSim" / "bin" / f"champsim_{pred}"
    env = {
        **dict(__import__("os").environ),
        "ANBA_EPOCH": str(epoch),
        "ANBA_PERSIST_STATE": "1",
        "ANBA_CHECKPOINT_PATH": str(ckpt),
        **extra,
    }
    cmd = [str(bin_path), f"--warmup-instructions={warmup}", f"--simulation-instructions={sim}", str(trace)]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        return None
    parsed = parse_log(proc.stdout)
    return parsed["branch_accuracy_pct"]


def main() -> int:
    split = json.loads((ROOT / "data/splits/online_repeating_v2.json").read_text())
    val = ROOT / "data/traces" / split["val_trace"]
    warmup = int(split["warmup"])
    sim = int(split["sim"])
    grid = split["hyperparam_grid"]
    ckpt_dir = ROOT / "results" / "checkpoints" / "tune"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    keys = ["ANBA_CHOOSER_HIGH", "ANBA_MODEL_MARGIN", "ANBA_CHOOSER_LOW", "ANBA_LEARN_SHIFT"]
    combos = list(
        itertools.product(
            grid["ANBA_CHOOSER_HIGH"],
            grid["ANBA_MODEL_MARGIN"],
            grid["ANBA_CHOOSER_LOW"],
            grid["ANBA_LEARN_SHIFT"],
        )
    )
    best = None
    results = []
    for ch, margin, low, lshift in combos:
        params = {
            "ANBA_CHOOSER_HIGH": str(ch),
            "ANBA_MODEL_MARGIN": str(margin),
            "ANBA_CHOOSER_LOW": str(low),
            "ANBA_LEARN_SHIFT": str(lshift),
        }
        g_ck = ckpt_dir / f"gshare_{ch}_{margin}_{low}_{lshift}.bin"
        h_ck = ckpt_dir / f"hybrid_{ch}_{margin}_{low}_{lshift}.bin"
        if g_ck.exists():
            g_ck.unlink()
        if h_ck.exists():
            h_ck.unlink()
        g1 = run_epoch("gshare_repeat", val, warmup, sim, 1, g_ck, {})
        g2 = run_epoch("gshare_repeat", val, warmup, sim, 2, g_ck, {}) if g1 is not None else None
        h1 = run_epoch("anba_gshare_online", val, warmup, sim, 1, h_ck, params)
        h2 = run_epoch("anba_gshare_online", val, warmup, sim, 2, h_ck, params) if h1 is not None else None
        if g2 is None or h2 is None:
            continue
        delta = h2 - g2
        row = {**params, "gshare_ep2": g2, "hybrid_ep2": h2, "delta_vs_gshare": delta}
        results.append(row)
        if best is None or delta > best["delta_vs_gshare"]:
            best = row

    out = {
        "val_trace": split["val_trace"],
        "grid_results": results,
        "best": best,
    }
    out_path = ROOT / "results" / "parsed" / "online_tune_val.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    if best:
        split["tuned_params"] = {
            "ANBA_CHOOSER_HIGH": int(best["ANBA_CHOOSER_HIGH"]),
            "ANBA_MODEL_MARGIN": int(best["ANBA_MODEL_MARGIN"]),
            "ANBA_CHOOSER_LOW": int(best["ANBA_CHOOSER_LOW"]),
            "ANBA_LEARN_SHIFT": int(best["ANBA_LEARN_SHIFT"]),
        }
        (ROOT / "data/splits/online_repeating_v2.json").write_text(json.dumps(split, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
