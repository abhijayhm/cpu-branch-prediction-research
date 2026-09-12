#!/usr/bin/env python3
"""Freeze spec_residual_v1.json: hard-domain trace split + residual curriculum metadata."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARD_DOMAIN = ROOT / "data" / "splits" / "spec_hard_domain_v1.json"
OUT = ROOT / "data" / "splits" / "spec_residual_v1.json"
SEED = 20260915


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if OUT.exists() and not args.force:
        print(f"{OUT} already exists (use --force to overwrite)")
        return 0
    if not HARD_DOMAIN.exists():
        print(f"missing parent split {HARD_DOMAIN}; run make hard-domain-splits first", file=__import__("sys").stderr)
        return 1

    parent = json.loads(HARD_DOMAIN.read_text())
    hd = parent["hard_domain_subset"]
    doc = {
        "name": "spec_residual_v1",
        "seed": SEED,
        "unit": "workload",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_catalog": parent.get("source_catalog"),
        "source_doi": parent.get("source_doi"),
        "parent_split": str(HARD_DOMAIN.relative_to(ROOT)),
        "rules": [
            "Same workload-level train/val/test traces as spec_hard_domain_v1.",
            "Never train or build residual CSVs from test traces.",
            "Residual CSVs emphasize bimodal mispredictions (predicted != taken).",
            "Agreement rows are mixed in at a calibrated fraction (default 15%).",
            "Supervised target remains true taken (not flip-of-bimodal).",
        ],
        "residual_curriculum": {
            "agreement_fraction_default": 0.15,
            "emphasis": "predicted != taken",
            "target_label": "taken",
            "data_dir": "data/extracted/residual",
            "builder": "tools/build_residual_dataset.py",
        },
        "residual_subset": hd,
        "hard_domain_subset": hd,
        "small_subset": parent.get("small_subset", hd),
        "bimodal_probe": parent.get("bimodal_probe", {}),
        "dataset_note": (
            "Residual training rows are a mix of bimodal disagreements and sampled "
            "agreements. Labels are always the true taken bit."
        ),
    }
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
