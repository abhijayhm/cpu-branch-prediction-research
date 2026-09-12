#!/usr/bin/env python3
"""Write a ChampSim JSON config that shares the frozen CPU model.

Only ooo_cpu[0].branch_predictor and executable_name change.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs" / "cpu" / "base.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictor", required=True, help="ChampSim branch/ directory name")
    parser.add_argument("--out", required=True)
    parser.add_argument("--executable-name", default=None)
    args = parser.parse_args()

    cfg = json.loads(BASE.read_text())
    cfg.pop("comment", None)
    cfg["ooo_cpu"][0]["branch_predictor"] = args.predictor
    cfg["executable_name"] = args.executable_name or f"champsim_{args.predictor}"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, indent=2) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
