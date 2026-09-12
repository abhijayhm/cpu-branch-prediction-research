#!/usr/bin/env python3
"""Export a trained tiny NN to the ANBA INT8 binary consumed by C++ ChampSim."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.models import REGISTRY  # noqa: E402

MAGIC = 0x414E4241
VERSION = 1


def quantize_linear(weight: np.ndarray, bias: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Symmetric INT8 weights; bias in accumulator units (bias_f / scale)."""
    max_abs = float(np.max(np.abs(weight))) if weight.size else 1.0
    scale = max_abs / 127.0 if max_abs > 0 else 1.0
    q = np.clip(np.round(weight / scale), -127, 127).astype(np.int8)
    b = np.clip(np.round(bias / scale), -2_000_000_000, 2_000_000_000).astype(np.int32)
    return q, b, scale


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--arch", default="nn_a")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("PyTorch required for export", file=sys.stderr)
        return 1

    ckpt_path = args.ckpt or (ROOT / "models" / f"{args.arch}.pt")
    if not ckpt_path.exists():
        print(f"missing checkpoint {ckpt_path}", file=sys.stderr)
        return 1
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = blob.get("arch", args.arch)
    n_in = int(blob.get("n_in", 32))
    model = REGISTRY[arch](n_in)
    model.load_state_dict(blob["state_dict"])
    model.eval()

    linears = [m for m in model.modules() if type(m).__name__ == "Linear"]
    out = args.out or (ROOT / "models" / "export" / f"{arch}_int8.bin")
    out.parent.mkdir(parents=True, exist_ok=True)

    layers_meta = []
    with out.open("wb") as f:
        f.write(struct.pack("<I", MAGIC))
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<I", int(model.arch_id)))
        f.write(struct.pack("<I", n_in))
        f.write(struct.pack("<I", len(linears)))
        for lin in linears:
            w = lin.weight.detach().cpu().numpy().astype(np.float32)
            b = lin.bias.detach().cpu().numpy().astype(np.float32)
            q, qb, scale = quantize_linear(w, b)
            f.write(struct.pack("<I", int(w.shape[1])))
            f.write(struct.pack("<I", int(w.shape[0])))
            f.write(struct.pack("<f", float(scale)))
            f.write(q.tobytes(order="C"))
            f.write(qb.tobytes(order="C"))
            layers_meta.append(
                {
                    "in": int(w.shape[1]),
                    "out": int(w.shape[0]),
                    "scale": float(scale),
                    "weight_minmax": [float(w.min()), float(w.max())],
                }
            )

    meta = {
        "path": str(out),
        "arch": arch,
        "arch_id": int(model.arch_id),
        "n_features": n_in,
        "n_layers": len(linears),
        "layers": layers_meta,
        "note": "C++ hidden activations are sign(); export matches that only approximately vs Tanh train.",
        "fabricated": False,
    }
    meta_path = out.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
