#!/usr/bin/env python3
"""Train NN-T on residual curriculum (ordered traces). Tune gate margin on val only."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.nn_t_model import CausalBranchTransformer  # noqa: E402
from ml.train_residual import resolve_residual_files, trace_stem  # noqa: E402
from ml.transformer_data import (  # noqa: E402
    build_sequences,
    read_instrumented_ordered,
    residual_row_mask,
)


def resolve_instrumented_files(split_doc: dict, part: str, input_dir: Path) -> list[Path]:
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
        cand = input_dir / f"{stem}.csv"
        if cand.exists():
            found.append(cand)
    return found


def stack_sequences(
    files: list[Path],
    input_dir: Path,
    agreement_fraction: float,
    cap: int | None,
    seed: int,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    tokens_all, labels_all = [], []
    for p in files:
        pc, _hist, taken, pred = read_instrumented_ordered(p)
        mask = residual_row_mask(taken, pred, agreement_fraction, cap, seed + hash(p.stem) % 10_000)
        batch = build_sequences(pc, taken, mask, seq_len)
        if len(batch.labels):
            tokens_all.append(batch.tokens)
            labels_all.append(batch.labels)
    if not tokens_all:
        return np.zeros((0, seq_len), dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.concatenate(tokens_all), np.concatenate(labels_all)


def simulate_bimodal_uncertain(pc: np.ndarray, taken: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """2-bit saturating counters; return (bimodal_pred, uncertain) per branch in order."""
    table: dict[int, int] = {}
    preds = np.zeros(len(pc), dtype=np.int64)
    uncertain = np.zeros(len(pc), dtype=bool)
    for i, (p, t) in enumerate(zip(pc, taken)):
        h = int(p) % 16381
        ctr = table.get(h, 0)  # matches ChampSim fwcounter default (0)
        preds[i] = 1 if ctr > 1 else 0
        uncertain[i] = ctr not in (0, 3)
        ctr = max(0, min(3, ctr + (1 if t else -1)))
        table[h] = ctr
    return preds, uncertain


def tune_margin(
    model,
    pc: np.ndarray,
    taken: np.ndarray,
    seq_len: int,
    margins: list[float],
) -> tuple[float, dict]:
    import torch

    from ml.transformer_data import build_sequences

    bimodal, uncertain = simulate_bimodal_uncertain(pc, taken)
    if not np.any(uncertain):
        return margins[0], {"val_gated_acc": 0.0, "note": "no uncertain branches on val"}

    # Full-trace ordered logits at every branch (for gate sim on uncertain rows).
    mask_all = np.ones(len(pc), dtype=bool)
    batch = build_sequences(pc, taken, mask_all, seq_len)
    with torch.no_grad():
        logits_all = model(torch.from_numpy(batch.tokens)).numpy()

    idx = np.flatnonzero(uncertain)
    best_m = margins[0]
    best_acc = -1.0
    best_stats: dict = {}
    for m in margins:
        nn_pred = (logits_all >= 0).astype(np.int64)
        override = np.abs(logits_all) >= m
        gated = bimodal.copy()
        gated[uncertain & override] = nn_pred[uncertain & override]
        acc = float((gated == taken).mean())
        if acc > best_acc:
            best_acc = acc
            best_m = m
            n_unc = int(len(idx))
            n_ov = int((uncertain & override).sum())
            ov_mask = uncertain & override
            prec = float((nn_pred[ov_mask] == taken[ov_mask]).mean()) if n_ov else 0.0
            best_stats = {
                "val_gated_acc": acc,
                "val_uncertain_branches": n_unc,
                "val_override_rate": round(n_ov / max(n_unc, 1), 4),
                "val_override_precision": round(prec, 4),
                "margin": m,
            }
    return best_m, best_stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=Path, default=ROOT / "data" / "splits" / "spec_residual_v1.json")
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data" / "extracted")
    parser.add_argument("--agreement-fraction", type=float, default=0.15)
    parser.add_argument("--cap", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=48)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--nlayers", type=int, default=2)
    parser.add_argument(
        "--export-out",
        type=Path,
        default=ROOT / "models" / "export" / "nn_transformer.bin",
    )
    args = parser.parse_args()

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("PyTorch required", file=sys.stderr)
        return 1

    split_doc = json.loads(args.splits.read_text())
    train_files = resolve_instrumented_files(split_doc, "train", args.input_dir)
    val_files = resolve_instrumented_files(split_doc, "val", args.input_dir)
    test_files = resolve_instrumented_files(split_doc, "test", args.input_dir)
    if test_files:
        print(f"refusing test files: {[p.name for p in test_files]}", file=sys.stderr)
    if not train_files or not val_files:
        print("missing instrumented train/val CSVs", file=sys.stderr)
        return 1

    x_train, y_train = stack_sequences(
        train_files, args.input_dir, args.agreement_fraction, args.cap, args.seed, args.seq_len
    )
    x_val, y_val = stack_sequences(
        val_files, args.input_dir, args.agreement_fraction, args.cap, args.seed, args.seq_len
    )
    print(f"nn-t train seqs={len(y_train)} val seqs={len(y_val)}")
    if len(y_train) < 100:
        print("too few training sequences", file=sys.stderr)
        return 1

    torch.manual_seed(args.seed)
    model = CausalBranchTransformer(
        seq_len=args.seq_len,
        d_model=args.d_model,
        nhead=args.nhead,
        nlayers=args.nlayers,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    ds = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(y_train.astype(np.float32)),
    )
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
        print(f"[nn_t] epoch {epoch+1}/{args.epochs} loss={total/max(n,1):.4f} val_acc={val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        val_logits = model(torch.from_numpy(x_val))
        val_acc = float(((val_logits > 0).numpy().astype(np.int64) == y_val).mean())
        train_logits = model(torch.from_numpy(x_train))
        train_acc = float(((train_logits > 0).numpy().astype(np.int64) == y_train).mean())

    beat_random = val_acc > 0.5
    if not beat_random:
        print("nn-t did not beat random on val", file=sys.stderr)
        return 3

    # Val margin tune (full ordered val trace for bimodal sim; labels at residual rows)
    val_pc_parts, val_taken_parts = [], []
    for vf in val_files:
        pc, _, taken, _ = read_instrumented_ordered(vf)
        val_pc_parts.append(pc)
        val_taken_parts.append(taken)
    val_pc_all = np.concatenate(val_pc_parts)
    val_taken_all = np.concatenate(val_taken_parts)

    margins = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
    best_margin, margin_stats = tune_margin(model, val_pc_all, val_taken_all, args.seq_len, margins)

    ckpt = ROOT / "models" / "nn_t.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "arch": "nn_t",
            "state_dict": model.state_dict(),
            "seq_len": args.seq_len,
            "d_model": args.d_model,
            "nhead": args.nhead,
            "nlayers": args.nlayers,
            "margin": best_margin,
        },
        ckpt,
    )

    import subprocess

    rc = subprocess.call(
        [
            sys.executable,
            str(ROOT / "ml" / "export_transformer.py"),
            "--ckpt",
            str(ckpt),
            "--out",
            str(args.export_out),
            "--margin",
            str(best_margin),
        ]
    )
    if rc != 0:
        return rc

    report = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": "residual_ordered",
        "arch": "nn_t",
        "model_name": model.name,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "train_acc": train_acc,
        "val_acc": val_acc,
        "random_acc": 0.5,
        "beat_random": beat_random,
        "seq_len": args.seq_len,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "nlayers": args.nlayers,
        "gate_margin_logit": best_margin,
        "gate_val_stats": margin_stats,
        "train_files": [p.name for p in train_files],
        "val_files": [p.name for p in val_files],
        "fabricated": False,
    }
    out_dir = ROOT / "results" / "parsed"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_residual_nn_t.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
