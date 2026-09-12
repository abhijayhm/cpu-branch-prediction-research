#!/usr/bin/env python3
"""Build disagreement-focused training CSVs from instrumented dumps.

Input rows: pc,history,taken,branch_type,predicted (from instrumented predictor).
Output rows: same schema, emphasizing predicted != taken (bimodal mispredictions),
with a calibrated mix of agreement rows so the model is not trained only on rare events.

Never reads test-split traces (refuses if asked).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def trace_stem(name: str) -> str:
    return name.replace(".champsimtrace.xz", "").replace(".champsimtrace", "")


def resolve_split_traces(split_doc: dict, part: str) -> list[str]:
    """Return trace filenames for train/val from residual or hard-domain subset."""
    for key in ("residual_subset", "hard_domain_subset", "small_subset"):
        sub = split_doc.get(key)
        if sub and "traces" in sub and part in sub["traces"]:
            return list(sub["traces"][part])
    raise KeyError(f"no traces.{part} in split document")


def read_instrumented(path: Path) -> tuple[list[dict], list[dict], dict]:
    disagreements: list[dict] = []
    agreements: list[dict] = []
    total = 0
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            taken = int(row["taken"], 0)
            predicted = int(row.get("predicted", taken), 0)
            rec = {
                "pc": row["pc"],
                "history": row["history"],
                "taken": str(taken),
                "branch_type": row.get("branch_type", "0"),
                "predicted": str(predicted),
            }
            if predicted != taken:
                disagreements.append(rec)
            else:
                agreements.append(rec)
    stats = {
        "source": str(path),
        "total_rows": total,
        "disagreement_rows": len(disagreements),
        "agreement_rows": len(agreements),
        "disagreement_pct": round(100.0 * len(disagreements) / max(total, 1), 2),
    }
    return disagreements, agreements, stats


def mix_rows(
    disagreements: list[dict],
    agreements: list[dict],
    agreement_fraction: float,
    cap: int | None,
    seed: int,
) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    n_dis = len(disagreements)
    if agreement_fraction <= 0.0:
        n_agree = 0
    elif n_dis == 0:
        n_agree = min(len(agreements), cap or len(agreements))
    else:
        target_agree = int(round(n_dis * agreement_fraction / max(1.0 - agreement_fraction, 1e-9)))
        n_agree = min(len(agreements), target_agree)
    picked_agree = rng.sample(agreements, n_agree) if n_agree else []
    mixed = list(disagreements) + picked_agree
    rng.shuffle(mixed)
    if cap is not None and len(mixed) > cap:
        mixed = mixed[:cap]
    mix_stats = {
        "disagreement_rows_used": n_dis,
        "agreement_rows_used": len(picked_agree),
        "agreement_fraction_target": agreement_fraction,
        "agreement_fraction_actual": round(len(picked_agree) / max(len(mixed), 1), 4),
        "output_rows": len(mixed),
    }
    return mixed, mix_stats


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["pc", "history", "taken", "branch_type", "predicted"]
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits",
        type=Path,
        default=ROOT / "data" / "splits" / "spec_residual_v1.json",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "data" / "extracted",
        help="directory with instrumented {stem}.csv files",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "extracted" / "residual",
    )
    parser.add_argument(
        "--agreement-fraction",
        type=float,
        default=0.15,
        help="fraction of output rows that are bimodal-agreement (predicted==taken)",
    )
    parser.add_argument("--cap", type=int, default=100_000, help="max rows per output CSV")
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()

    if not args.splits.exists():
        print(f"missing split file {args.splits}", file=sys.stderr)
        return 1

    split_doc = json.loads(args.splits.read_text())
    test_traces = set(resolve_split_traces(split_doc, "test"))
    parts = ("train", "val")
    per_file: list[dict] = []
    missing: list[str] = []

    for part in parts:
        for trace in resolve_split_traces(split_doc, part):
            if trace in test_traces:
                print(f"refusing to build residual data from test trace {trace}", file=sys.stderr)
                return 1
            stem = trace_stem(trace)
            src = args.input_dir / f"{stem}.csv"
            if not src.exists():
                missing.append(str(src))
                continue
            disagreements, agreements, src_stats = read_instrumented(src)
            mixed, mix_stats = mix_rows(
                disagreements,
                agreements,
                args.agreement_fraction,
                args.cap,
                args.seed + hash(stem) % 10_000,
            )
            out_path = args.out_dir / f"{stem}.csv"
            write_csv(out_path, mixed)
            per_file.append(
                {
                    "split": part,
                    "trace": trace,
                    "output": str(out_path.relative_to(ROOT)),
                    **src_stats,
                    **mix_stats,
                }
            )
            print(
                f"{part} {stem}: {src_stats['disagreement_pct']}% disagree "
                f"-> {mix_stats['output_rows']} rows "
                f"(agree mix {mix_stats['agreement_fraction_actual']:.2%})"
            )

    if missing:
        print(f"missing instrumented CSVs: {missing}", file=sys.stderr)
        return 1
    if not per_file:
        print("no residual datasets built", file=sys.stderr)
        return 1

    total_out = sum(r["output_rows"] for r in per_file)
    total_dis = sum(r["disagreement_rows"] for r in per_file)
    report = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "splits": str(args.splits.relative_to(ROOT)),
        "input_dir": str(args.input_dir.relative_to(ROOT)),
        "out_dir": str(args.out_dir.relative_to(ROOT)),
        "agreement_fraction_target": args.agreement_fraction,
        "cap_per_file": args.cap,
        "seed": args.seed,
        "curriculum": (
            "Emphasize bimodal mispredictions (predicted != taken) with a calibrated "
            f"agreement mix ({args.agreement_fraction:.0%} target) so training is not "
            "dominated by rare disagreement events only."
        ),
        "files": per_file,
        "totals": {
            "output_rows": total_out,
            "disagreement_rows_in_source": total_dis,
            "mean_disagreement_pct_in_source": round(
                sum(r["disagreement_pct"] for r in per_file) / len(per_file), 2
            ),
        },
        "fabricated": False,
    }
    out_report = ROOT / "results" / "parsed" / "residual_dataset_stats.json"
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
