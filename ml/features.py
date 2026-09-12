"""Shared feature packing for train/export. Must match predictor/common/anba_int8.h."""

from __future__ import annotations

import numpy as np

N_FEATURES = 32
PC_BITS = 16
HIST_BITS = 16


def pack_features(pc: np.ndarray, history: np.ndarray) -> np.ndarray:
    pc = np.asarray(pc, dtype=np.uint64) >> np.uint64(2)
    history = np.asarray(history, dtype=np.uint32)
    x = np.empty((pc.shape[0], N_FEATURES), dtype=np.float32)
    for i in range(PC_BITS):
        bit = ((pc >> np.uint64(i)) & np.uint64(1)).astype(np.float32)
        x[:, i] = bit * 2.0 - 1.0
    for i in range(HIST_BITS):
        bit = ((history >> np.uint32(i)) & np.uint32(1)).astype(np.float32)
        x[:, PC_BITS + i] = bit * 2.0 - 1.0
    return x
