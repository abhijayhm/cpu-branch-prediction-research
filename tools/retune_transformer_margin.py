#!/usr/bin/env python3
"""Re-tune NN-T gate margin on val only (ChampSim-matched bimodal init=0)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.nn_t_model import CausalBranchTransformer  # noqa: E402
from ml.train_residual_transformer import simulate_bimodal_uncertain, tune_margin  # noqa: E402
from ml.transformer_data import build_sequences, read_instrumented_ordered  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=ROOT / "models" / "nn_t.pt")
    parser.add_argument("--val-csv", type=Path, default=ROOT / "data" / "extracted" / "631.deepsjeng_s-928B.csv")
    parser.add_argument("--export-out", type=Path, default=ROOT / "models" / "export" / "nn_transformer.bin")
    args = parser.parse_args()

    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = CausalBranchTransformer(
        seq_len=int(blob["seq_len"]),
        d_model=int(blob["d_model"]),
        nhead=int(blob["nhead"]),
        nlayers=int(blob["nlayers"]),
    )
    model.load_state_dict(blob["state_dict"])
    model.eval()

    pc, taken, _, _ = read_instrumented_ordered(args.val_csv)
    margins = [4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0, 32.0]
    best_m, stats = tune_margin(model, pc, taken, int(blob["seq_len"]), margins)
    # Prefer margins that do not reduce gated acc below bimodal on val replay (warmup skipped).
    from ml.transformer_data import branch_token, current_token, VOCAB_CURRENT_OFFSET

    PRIME = 16381
    warmup_skip = min(15000, len(pc) // 3)

    def replay_acc(margin: float) -> float:
        table: dict[int, int] = {}
        ring: list[int] = []
        g_corr = 0
        for i in range(len(pc)):
            ip = int(pc[i])
            h = ip % PRIME
            ctr = table.get(h, 0)
            conv = ctr > 1
            pred = conv
            if ctr not in (0, 3):
                hist = ring[-(int(blob["seq_len"]) - 1) :]
                pad_n = int(blob["seq_len"]) - 1 - len(hist)
                toks = [VOCAB_CURRENT_OFFSET] * pad_n + hist + [current_token(ip)]
                logit = float(model(torch.tensor([toks])).item())
                if abs(logit) >= margin:
                    pred = logit >= 0
            if i >= warmup_skip:
                g_corr += int(pred == int(taken[i]))
            ctr = max(0, min(3, ctr + (1 if taken[i] else -1)))
            table[h] = ctr
            ring.append(branch_token(ip, int(taken[i])))
            if len(ring) > int(blob["seq_len"]) - 1:
                ring.pop(0)
        return g_corr / max(len(pc) - warmup_skip, 1)

    bimodal_acc = replay_acc(1e9)
    safe = [m for m in margins if replay_acc(m) + 1e-9 >= bimodal_acc]
    if safe:
        best_m = max(safe)
        stats = {
            "val_gated_acc": replay_acc(best_m),
            "val_bimodal_acc": bimodal_acc,
            "val_safe_margins": safe,
            "margin": best_m,
            "warmup_skip": warmup_skip,
        }

    rc = subprocess.call(
        [
            sys.executable,
            str(ROOT / "ml" / "export_transformer.py"),
            "--ckpt",
            str(args.ckpt),
            "--out",
            str(args.export_out),
            "--margin",
            str(best_m),
        ]
    )
    if rc != 0:
        return rc

    out = {
        "gate_margin_logit": best_m,
        "gate_val_stats": stats,
        "val_csv": str(args.val_csv),
        "fabricated": False,
    }
    path = ROOT / "results" / "parsed" / "train_residual_nn_t_margin.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
