#!/usr/bin/env python3
"""Parity: PyTorch vs C++ logits on shared token sequences from val traces."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.nn_t_model import CausalBranchTransformer  # noqa: E402
from ml.transformer_data import build_sequences, read_instrumented_ordered  # noqa: E402

CPP = ROOT / "tools" / "cpp_transformer_logit"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=ROOT / "models" / "nn_t.pt")
    parser.add_argument("--bin", type=Path, default=ROOT / "models" / "export" / "nn_transformer.bin")
    parser.add_argument("--csv", type=Path, default=ROOT / "data" / "extracted" / "631.deepsjeng_s-928B.csv")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--tol", type=float, default=0.05)
    args = parser.parse_args()

    if not CPP.exists():
        print("build tools/cpp_transformer_logit first", file=sys.stderr)
        return 1

    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = CausalBranchTransformer(
        seq_len=int(blob["seq_len"]),
        d_model=int(blob["d_model"]),
        nhead=int(blob["nhead"]),
        nlayers=int(blob["nlayers"]),
    )
    model.load_state_dict(blob["state_dict"])
    model.eval()

    pc, taken, _, _ = read_instrumented_ordered(args.csv)
    batch = build_sequences(pc, taken, np.ones(len(pc), dtype=bool), int(blob["seq_len"]))
    n = min(args.n, len(batch.tokens))
    worst = 0.0
    bad = 0
    env = {"ANBA_MODEL_PATH": str(args.bin)}
    for i in range(n):
        toks = batch.tokens[i].tolist()
        with torch.no_grad():
            ref = float(model(torch.from_numpy(batch.tokens[i : i + 1])).item())
        cpp = float(
            subprocess.check_output([str(CPP)], input=" ".join(map(str, toks)), text=True, env=env).strip()
        )
        err = abs(ref - cpp)
        worst = max(worst, err)
        if err > args.tol:
            bad += 1
    report = {
        "n": n,
        "tol": args.tol,
        "mismatches": bad,
        "max_abs_err": worst,
        "ok": bad == 0,
        "fabricated": False,
    }
    print(__import__("json").dumps(report, indent=2))
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
