#!/usr/bin/env python3
"""Report A1–A11 from real artifacts only. Never invents metrics."""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def exists(*parts: str) -> bool:
    return (ROOT.joinpath(*parts)).exists()


def load_json(rel: str) -> dict | None:
    p = ROOT / rel
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def status(ok: bool, detail: str) -> dict:
    return {"pass": bool(ok), "detail": detail}


def main() -> int:
    versions = load_json("data/metadata/software_versions.json") or {}
    pin = ((versions.get("champsim") or {}).get("commit")) or ""
    champsim_head = ""
    cs = ROOT / "champsim" / "ChampSim"
    if (cs / ".git").exists():
        import subprocess

        champsim_head = subprocess.check_output(["git", "-C", str(cs), "rev-parse", "HEAD"], text=True).strip()

    traces_csv = ROOT / "data" / "metadata" / "traces.csv"
    n_catalog = 0
    n_downloaded = 0
    n_sha = 0
    if traces_csv.exists():
        with traces_csv.open() as f:
            rows = list(csv.DictReader(f))
        n_catalog = len(rows)
        n_downloaded = sum(1 for r in rows if r.get("downloaded") == "yes")
        n_sha = sum(1 for r in rows if r.get("sha256"))

    parsed = list((ROOT / "results" / "parsed").glob("*.json")) if (ROOT / "results" / "parsed").exists() else []
    baseline = None
    for p in sorted((ROOT / "results" / "parsed").glob("latest_bimodal.json")):
        baseline = json.loads(p.read_text())
    train = load_json("results/parsed/train_nn_a.json") or load_json("results/parsed/train_nn_c.json")
    inst = load_json("results/parsed/instrumentation_verify.json")
    splits = load_json("data/splits/spec_v1.json")
    export = list((ROOT / "models" / "export").glob("*_int8.bin")) if (ROOT / "models" / "export").exists() else []

    a = {}
    a["A1"] = status(
        exists("configs/cpu/base.json") and exists("Makefile") and exists("requirements.txt"),
        "scaffold files present",
    )
    a["A2"] = status(
        exists("environment.yml") and exists("Dockerfile") and exists("LICENSE"),
        "env files present (conda/docker/license)",
    )
    a["A3"] = status(
        bool(pin) and (not champsim_head or champsim_head == pin),
        f"pin={pin or 'missing'} local={champsim_head or 'not cloned'}",
    )
    a["A4"] = status(
        n_catalog > 0 and n_downloaded >= 1 and n_sha >= 1,
        f"catalog={n_catalog} downloaded={n_downloaded} sha256_filled={n_sha}",
    )
    bin_bimodal = ROOT / "champsim" / "ChampSim" / "bin" / "champsim_bimodal"
    alt_bin = ROOT / "champsim" / "ChampSim" / "bin" / "champsim"
    a["A5"] = status(bin_bimodal.exists() or alt_bin.exists(), f"bimodal_bin={bin_bimodal.exists()} default_bin={alt_bin.exists()}")
    base_ok = bool(baseline and baseline.get("parse_ok") and baseline.get("metrics", {}).get("ipc") not in (None, "unavailable"))
    a["A6"] = status(
        base_ok,
        json.dumps(baseline.get("metrics") if baseline else {"metrics": "unavailable"}, sort_keys=True),
    )
    a["A7"] = status(bool(inst and inst.get("verification", {}).get("ok")), inst["verification"]["notes"] if inst else "not run")
    a["A8"] = status(
        bool(splits and splits.get("seed") == 20260912 and splits.get("unit") == "workload"),
        f"seed={splits.get('seed') if splits else 'missing'} unit={splits.get('unit') if splits else 'missing'}",
    )
    a["A9"] = status(
        bool(train and train.get("beat_random") is True),
        f"val_acc={train.get('val_acc') if train else 'unavailable'} beat_random={train.get('beat_random') if train else 'unavailable'}",
    )
    nn_bin = ROOT / "champsim" / "ChampSim" / "bin" / "champsim_nn_frozen"
    a["A10"] = status(bool(export) and nn_bin.exists(), f"int8_export={bool(export)} nn_bin={nn_bin.exists()}")
    hy_bin = ROOT / "champsim" / "ChampSim" / "bin" / "champsim_anba_hybrid"
    a["A11"] = status(
        hy_bin.exists() and exists("docs/IMPLEMENTATION_NOTES.md"),
        f"hybrid_bin={hy_bin.exists()} notes={exists('docs/IMPLEMENTATION_NOTES.md')}",
    )

    doc = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "criteria": a,
        "passed": sum(1 for v in a.values() if v["pass"]),
        "total": len(a),
        "fabricated": False,
    }
    out = ROOT / "results" / "parsed" / "acceptance.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n")
    print(json.dumps(doc, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
