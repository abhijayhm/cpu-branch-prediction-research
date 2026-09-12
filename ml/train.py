#!/usr/bin/env python3
"""Train a tiny NN on frozen train split only. Gate: must beat random on val.

Random = 50% coin-flip accuracy. Majority-class accuracy is reported but is
not the acceptance gate. Never reads the test split.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.features import pack_features  # noqa: E402
from ml.models import REGISTRY  # noqa: E402


def load_csv(path: Path, cap: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pcs, hists, ys = [], [], []
    with path.open() as f:
        header = f.readline()
        if "pc" not in header:
            f.seek(0)
        for i, line in enumerate(f):
            if cap is not None and i >= cap:
                break
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            pcs.append(int(parts[0], 0))
            hists.append(int(parts[1], 0))
            ys.append(int(parts[2], 0))
    return np.asarray(pcs, dtype=np.uint64), np.asarray(hists, dtype=np.uint32), np.asarray(ys, dtype=np.int64)


def resolve_split_files(split_doc: dict, part: str) -> list[Path]:
    names = split_doc["small_subset"]["traces"][part]
    extracted = ROOT / "data" / "extracted"
    found = []
    for name in names:
        stem = name.replace(".champsimtrace.xz", "").replace(".champsimtrace", "")
        cand = extracted / f"{stem}.csv"
        if cand.exists():
            found.append(cand)
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="nn_a", choices=sorted(REGISTRY))
    parser.add_argument("--splits", type=Path, default=ROOT / "data" / "splits" / "spec_v1.json")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--cap", type=int, default=100_000, help="max rows per CSV")
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, default=5, help="early-stop epochs without val improvement")
    parser.add_argument("--optimizer", choices=("adam", "sgd"), default=None)
    args = parser.parse_args()

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("PyTorch is required for training/export only. pip install -r requirements.txt", file=sys.stderr)
        return 1

    split_doc = json.loads(args.splits.read_text())
    train_files = resolve_split_files(split_doc, "train")
    val_files = resolve_split_files(split_doc, "val")
    test_files = resolve_split_files(split_doc, "test")
    if test_files:
        print(f"refusing to load test files during training: {[p.name for p in test_files]}", file=sys.stderr)
    if not train_files:
        print("no train CSVs under data/extracted for the frozen train split", file=sys.stderr)
        return 1
    if not val_files:
        print("no val CSVs under data/extracted; cannot enforce the random-baseline gate", file=sys.stderr)
        return 1

    def stack(files: list[Path]) -> tuple[np.ndarray, np.ndarray]:
        pcs, hists, ys = [], [], []
        for p in files:
            a, b, c = load_csv(p, args.cap)
            pcs.append(a)
            hists.append(b)
            ys.append(c)
        pc = np.concatenate(pcs)
        hist = np.concatenate(hists)
        y = np.concatenate(ys)
        return pack_features(pc, hist), y

    x_train, y_train = stack(train_files)
    x_val, y_val = stack(val_files)
    print(f"train rows={len(y_train)} taken_frac={float(y_train.mean()):.4f} files={[p.name for p in train_files]}")
    print(f"val rows={len(y_val)} taken_frac={float(y_val.mean()):.4f} files={[p.name for p in val_files]}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = REGISTRY[args.arch](x_train.shape[1])
    device = torch.device("cpu")
    model.to(device)
    # Cross-workload val (roms→fotonik3d) needs SGD+weight decay for all tiny nets;
    # Adam overfits on NN-A/NN-B hidden layers (see docs/IMPLEMENTATION_NOTES.md).
    opt_name = args.optimizer or "sgd"
    if opt_name == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.0)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.float32)))
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True)

    best_val = -1.0
    best_state = None
    stale = 0
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        n = 0
        for xb, yb in loader:
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(yb)
            n += len(yb)
        model.eval()
        with torch.no_grad():
            val_logits = model(torch.from_numpy(x_val))
            val_pred = (val_logits > 0).numpy().astype(np.int64)
            val_acc = float((val_pred == y_val).mean())
        print(f"epoch {epoch+1}/{args.epochs} train_loss={total/max(n,1):.4f} val_acc={val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stop at epoch {epoch+1} best_val={best_val:.4f}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        val_logits = model(torch.from_numpy(x_val))
        val_pred = (val_logits > 0).numpy().astype(np.int64)
        val_acc = float((val_pred == y_val).mean())
        train_logits = model(torch.from_numpy(x_train))
        train_acc = float(((train_logits > 0).numpy().astype(np.int64) == y_train).mean())
    random_acc = 0.5
    majority_acc = float(max(y_val.mean(), 1.0 - y_val.mean()))
    beat_random = val_acc > random_acc
    report = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "arch": args.arch,
        "model_name": model.name,
        "seed": args.seed,
        "splits": str(args.splits),
        "train_files": [p.name for p in train_files],
        "val_files": [p.name for p in val_files],
        "test_files_used": [],
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "train_acc": train_acc,
        "val_acc": val_acc,
        "random_acc": random_acc,
        "val_majority_acc": majority_acc,
        "beat_random": beat_random,
        "fabricated": False,
    }
    out_dir = ROOT / "results" / "parsed"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"train_{args.arch}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    ckpt = ROOT / "models" / f"{args.arch}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"arch": args.arch, "state_dict": model.state_dict(), "n_in": x_train.shape[1]}, ckpt)
    print(json.dumps(report, indent=2))
    if not beat_random:
        print("STOP: validation accuracy did not beat random (0.5).", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
