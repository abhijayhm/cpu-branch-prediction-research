"""Ordered trace sequences for NN-T (residual curriculum, no shuffle)."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ml.train import load_csv  # noqa: F401 — re-export cap loader


PC_HASH_BITS = 11
VOCAB_BRANCH = (1 << PC_HASH_BITS) * 2
VOCAB_CURRENT_OFFSET = VOCAB_BRANCH  # pc-only tokens (unknown taken)
VOCAB_SIZE = VOCAB_CURRENT_OFFSET + (1 << PC_HASH_BITS)


def pc_hash(pc: int) -> int:
    return int((int(pc) >> 2) & ((1 << PC_HASH_BITS) - 1))


def branch_token(pc: int, taken: int) -> int:
    return (pc_hash(pc) << 1) | (int(taken) & 1)


def current_token(pc: int) -> int:
    return VOCAB_CURRENT_OFFSET + pc_hash(pc)


def read_instrumented_ordered(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return pc, history, taken, predicted arrays in trace order."""
    pcs, hists, ys, preds = [], [], [], []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pcs.append(int(row["pc"], 0))
            hists.append(int(row["history"], 0))
            ys.append(int(row["taken"], 0))
            preds.append(int(row.get("predicted", row["taken"]), 0))
    return (
        np.asarray(pcs, dtype=np.uint64),
        np.asarray(hists, dtype=np.uint32),
        np.asarray(ys, dtype=np.int64),
        np.asarray(preds, dtype=np.int64),
    )


def residual_row_mask(
    taken: np.ndarray,
    predicted: np.ndarray,
    agreement_fraction: float,
    cap: int | None,
    seed: int,
) -> np.ndarray:
    """Same row selection as tools/build_residual_dataset.mix_rows, indices in order."""
    n = len(taken)
    disagree = predicted != taken
    agree_idx = np.flatnonzero(~disagree)
    dis_idx = np.flatnonzero(disagree)
    rng = random.Random(seed)
    n_dis = len(dis_idx)
    if agreement_fraction <= 0.0:
        n_agree = 0
    elif n_dis == 0:
        n_agree = min(len(agree_idx), cap or len(agree_idx))
    else:
        target_agree = int(round(n_dis * agreement_fraction / max(1.0 - agreement_fraction, 1e-9)))
        n_agree = min(len(agree_idx), target_agree)
    picked_agree = set(rng.sample(list(agree_idx), n_agree)) if n_agree else set()
    mask = disagree.copy()
    for i in picked_agree:
        mask[i] = True
    selected = np.flatnonzero(mask)
    if cap is not None and len(selected) > cap:
        selected = selected[:cap]
    out = np.zeros(n, dtype=bool)
    out[selected] = True
    return out


@dataclass
class SequenceBatch:
    tokens: np.ndarray  # int64 [N, L]
    labels: np.ndarray  # int64 [N]
    pcs: np.ndarray  # uint64 [N] — branch PC at prediction site


def build_sequences(
    pc: np.ndarray,
    taken: np.ndarray,
    train_mask: np.ndarray,
    seq_len: int,
) -> SequenceBatch:
    """Causal windows: past (pc,taken) tokens + current-PC unknown token; label = taken[i]."""
    n = len(pc)
    rows: list[np.ndarray] = []
    labels: list[int] = []
    pcs_out: list[int] = []
    history: list[int] = []
    for i in range(n):
        if train_mask[i]:
            window = history[-(seq_len - 1) :] if seq_len > 1 else []
            tok = list(window) + [current_token(int(pc[i]))]
            if len(tok) < seq_len:
                pad = [VOCAB_CURRENT_OFFSET] * (seq_len - len(tok))
                tok = pad + tok
            else:
                tok = tok[-seq_len:]
            rows.append(np.asarray(tok, dtype=np.int64))
            labels.append(int(taken[i]))
            pcs_out.append(int(pc[i]))
        history.append(branch_token(int(pc[i]), int(taken[i])))
        if len(history) > seq_len:
            history.pop(0)
    if not rows:
        return SequenceBatch(
            np.zeros((0, seq_len), dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.uint64),
        )
    return SequenceBatch(
        np.stack(rows, axis=0),
        np.asarray(labels, dtype=np.int64),
        np.asarray(pcs_out, dtype=np.uint64),
    )
