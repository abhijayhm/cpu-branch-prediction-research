#!/usr/bin/env python3
"""Train residual specialist NNs on disagreement-focused CSVs.

Trains NN-A/B/C, picks the best validation architecture (must beat random),
exports INT8 to models/export/nn_residual_int8.bin. Never reads test traces.
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

from ml.export_int8 import main as export_main  # noqa: E402
from ml.features import pack_features  # noqa: E402
from ml.models import REGISTRY  # noqa: E402
from ml.train import load_csv  # noqa: E402


def trace_stem(name: str) -> str:
    return name.replace(".champsimtrace.xz", "").replace(".champsimtrace", "")


def resolve_residual_files(split_doc: dict, part: str, data_dir: Path) -> list[Path]:
    for key in ("residual_subset", "hard_domain_subset", "small_subset"):
        sub = split_doc.get(key)
        if sub and "traces" in sub and part in sub["traces"]:
            names = sub["traces"][part]
            break
    else:
        return []
    found = []
    for name in names:
        stem = trace_stem(name)
        cand = data_dir / f"{stem}.csv"
        if cand.exists():
            found.append(cand)
    return found


def train_one_arch(
    arch: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict, object]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(args.seed)
    model = REGISTRY[arch](x_train.shape[1])
    opt_name = args.optimizer or "sgd"
    if opt_name == "sgd":
        opt = torch.optim.SGD(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.0
        )
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
        print(f"[{arch}] epoch {epoch+1}/{args.epochs} train_loss={total/max(n,1):.4f} val_acc={val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"[{arch}] early stop at epoch {epoch+1} best_val={best_val:.4f}")
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
    beat_random = val_acc > random_acc
    report = {
        "arch": arch,
        "model_name": model.name,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "train_acc": train_acc,
        "val_acc": val_acc,
        "random_acc": random_acc,
        "beat_random": beat_random,
    }
    return report, model


def stack(files: list[Path], cap: int | None) -> tuple[np.ndarray, np.ndarray]:
    pcs, hists, ys = [], [], []
    for p in files:
        a, b, c = load_csv(p, cap)
        pcs.append(a)
        hists.append(b)
        ys.append(c)
    pc = np.concatenate(pcs)
    hist = np.concatenate(hists)
    y = np.concatenate(ys)
    return pack_features(pc, hist), y


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=Path, default=ROOT / "data" / "splits" / "spec_residual_v1.json")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "extracted" / "residual",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--cap", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--optimizer", choices=("adam", "sgd"), default=None)
    parser.add_argument("--archs", nargs="+", default=["nn_a", "nn_b", "nn_c"])
    parser.add_argument(
        "--export-out",
        type=Path,
        default=ROOT / "models" / "export" / "nn_residual_int8.bin",
    )
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("PyTorch required", file=sys.stderr)
        return 1

    split_doc = json.loads(args.splits.read_text())
    train_files = resolve_residual_files(split_doc, "train", args.data_dir)
    val_files = resolve_residual_files(split_doc, "val", args.data_dir)
    test_files = resolve_residual_files(split_doc, "test", args.data_dir)
    if test_files:
        print(f"refusing test files: {[p.name for p in test_files]}", file=sys.stderr)
    if not train_files or not val_files:
        print("missing residual train/val CSVs; run build_residual_dataset first", file=sys.stderr)
        return 1

    x_train, y_train = stack(train_files, args.cap)
    x_val, y_val = stack(val_files, args.cap)
    print(
        f"residual train rows={len(y_train)} taken_frac={float(y_train.mean()):.4f} "
        f"files={[p.name for p in train_files]}"
    )
    print(
        f"residual val rows={len(y_val)} taken_frac={float(y_val.mean()):.4f} "
        f"files={[p.name for p in val_files]}"
    )

    arch_reports: list[dict] = []
    best_arch = None
    best_val = -1.0
    best_model = None
    for arch in args.archs:
        if arch not in REGISTRY:
            print(f"unknown arch {arch}", file=sys.stderr)
            continue
        report, model = train_one_arch(arch, x_train, y_train, x_val, y_val, args)
        report_full = {
            "schema_version": 1,
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "dataset": "residual",
            "splits": str(args.splits),
            "data_dir": str(args.data_dir),
            "train_files": [p.name for p in train_files],
            "val_files": [p.name for p in val_files],
            "fabricated": False,
            **report,
        }
        out_dir = ROOT / "results" / "parsed"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"train_residual_{arch}.json").write_text(json.dumps(report_full, indent=2) + "\n")
        arch_reports.append(report_full)
        if report["beat_random"] and report["val_acc"] > best_val:
            best_val = report["val_acc"]
            best_arch = arch
            best_model = model

    if best_arch is None or best_model is None:
        print("no residual arch beat random on validation", file=sys.stderr)
        return 3

    ckpt = ROOT / "models" / f"residual_{best_arch}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"arch": best_arch, "state_dict": best_model.state_dict(), "n_in": x_train.shape[1]},
        ckpt,
    )
    args.export_out.parent.mkdir(parents=True, exist_ok=True)
    import subprocess

    rc = subprocess.call(
        [
            sys.executable,
            str(ROOT / "ml" / "export_int8.py"),
            "--arch",
            best_arch,
            "--ckpt",
            str(ckpt),
            "--out",
            str(args.export_out),
        ]
    )
    if rc != 0:
        return rc

    summary = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "best_arch": best_arch,
        "best_val_acc": best_val,
        "export_path": str(args.export_out.relative_to(ROOT)),
        "arch_reports": arch_reports,
        "fabricated": False,
    }
    summary_path = ROOT / "results" / "parsed" / "train_residual_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
