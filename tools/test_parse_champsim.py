#!/usr/bin/env python3
"""Parser unit test using a synthetic ChampSim-shaped log (not a sim result)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_champsim import parse_log

SAMPLE = """
Heartbeat CPU 0 instructions: 100000 cycles: 80000 heartbeat IPC: 1.25
CPU 0 cumulative IPC: 1.234567 instructions: 200000 cycles: 162000
CPU 0 Branch Prediction Accuracy: 94.12% MPKI: 5.43 Average ROB Occupancy at Mispredict: 12.0
"""


def main() -> int:
    p = parse_log(SAMPLE)
    assert p["parse_ok"]
    assert abs(p["ipc"] - 1.234567) < 1e-9
    assert p["instructions"] == 200000
    assert p["cycles"] == 162000
    assert abs(p["branch_accuracy_pct"] - 94.12) < 1e-9
    assert abs(p["branch_mpki"] - 5.43) < 1e-9
    empty = parse_log("no metrics here")
    assert empty["parse_ok"] is False
    assert empty["ipc"] is None
    print("parse_champsim unit test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
