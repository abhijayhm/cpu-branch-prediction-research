#!/usr/bin/env python3
"""Export NN-T float weights for C++ inference (ANBA transformer binary v2)."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.nn_t_model import CausalBranchTransformer  # noqa: E402
from ml.transformer_data import VOCAB_SIZE  # noqa: E402

MAGIC = 0x414E4254  # 'ANBT'
VERSION = 2


def write_tensor(f, name: str, arr) -> dict:
    import numpy as np

    a = np.asarray(arr, dtype=np.float32)
    name_b = name.encode("utf-8")
    f.write(struct.pack("<I", len(name_b)))
    f.write(name_b)
    f.write(struct.pack("<I", a.ndim))
    for d in a.shape:
        f.write(struct.pack("<I", int(d)))
    f.write(a.tobytes(order="C"))
    return {"name": name, "shape": list(a.shape)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "export" / "nn_transformer.bin")
    parser.add_argument("--margin", type=float, default=1.0)
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("PyTorch required", file=sys.stderr)
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

    tensors_meta = []
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        f.write(struct.pack("<I", MAGIC))
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<I", int(model.arch_id)))
        f.write(struct.pack("<I", int(model.seq_len)))
        f.write(struct.pack("<I", int(model.d_model)))
        f.write(struct.pack("<I", int(model.nhead)))
        f.write(struct.pack("<I", int(model.nlayers)))
        f.write(struct.pack("<I", VOCAB_SIZE))
        f.write(struct.pack("<f", float(args.margin)))
        state = model.state_dict()
        f.write(struct.pack("<I", len(state)))
        for name, tensor in state.items():
            tensors_meta.append(write_tensor(f, name, tensor.detach().cpu().numpy()))

    meta = {
        "path": str(out),
        "arch": "nn_t",
        "arch_id": model.arch_id,
        "seq_len": model.seq_len,
        "d_model": model.d_model,
        "nhead": model.nhead,
        "nlayers": model.nlayers,
        "vocab_size": VOCAB_SIZE,
        "margin": float(args.margin),
        "tensors": tensors_meta,
        "fabricated": False,
    }
    meta_path = out.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
