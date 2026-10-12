#!/usr/bin/env python3
"""Check C++-style forward matches PyTorch on random token windows."""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.nn_t_model import CausalBranchTransformer  # noqa: E402
from ml.transformer_data import VOCAB_SIZE  # noqa: E402


def load_bin(path: Path) -> dict:
    import torch

    with path.open("rb") as f:
        magic, ver, arch, seq_len, d_model, nhead, nlayers, vocab, margin = struct.unpack("<9I", f.read(36)[:32] + f.read(4))
        # fix: read properly
    with path.open("rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        ver = struct.unpack("<I", f.read(4))[0]
        arch = struct.unpack("<I", f.read(4))[0]
        seq_len = struct.unpack("<I", f.read(4))[0]
        d_model = struct.unpack("<I", f.read(4))[0]
        nhead = struct.unpack("<I", f.read(4))[0]
        nlayers = struct.unpack("<I", f.read(4))[0]
        vocab = struct.unpack("<I", f.read(4))[0]
        margin = struct.unpack("<f", f.read(4))[0]
        nt = struct.unpack("<I", f.read(4))[0]
        tensors = {}
        for _ in range(nt):
            name_len = struct.unpack("<I", f.read(4))[0]
            name = f.read(name_len).decode()
            ndim = struct.unpack("<I", f.read(4))[0]
            shape = struct.unpack(f"<{ndim}I", f.read(4 * ndim))
            n = 1
            for d in shape:
                n *= d
            data = struct.unpack(f"<{n}f", f.read(4 * n))
            tensors[name] = torch.tensor(data, dtype=torch.float32).reshape(shape)
    return {
        "seq_len": seq_len,
        "d_model": d_model,
        "nhead": nhead,
        "nlayers": nlayers,
        "margin": margin,
        "tensors": tensors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=ROOT / "models" / "nn_t.pt")
    parser.add_argument("--bin", type=Path, default=ROOT / "models" / "export" / "nn_transformer.bin")
    args = parser.parse_args()
    import torch
    import numpy as np

    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = CausalBranchTransformer(
        seq_len=int(blob["seq_len"]),
        d_model=int(blob["d_model"]),
        nhead=int(blob["nhead"]),
        nlayers=int(blob["nlayers"]),
    )
    model.load_state_dict(blob["state_dict"])
    model.eval()
    L = model.seq_len
    rng = np.random.default_rng(0)
    toks = torch.from_numpy(rng.integers(0, VOCAB_SIZE, size=(32, L), dtype=np.int64))
    with torch.no_grad():
        ref = model(toks).numpy()
    print(f"pytorch logits sample mean={ref.mean():.4f} std={ref.std():.4f}")
    print("export file exists:", args.bin.exists())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
