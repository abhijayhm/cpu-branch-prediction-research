#!/usr/bin/env python3
"""Download official SPEC CPU2017 ChampSim traces from Zenodo 10960004.

Never invents files or checksums. SHA256 is written only after a real download.
The full catalog (95 files) is written to data/metadata/traces.csv from the
Zenodo API; only a small preset is fetched unless --all is given.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ZENODO_API = "https://zenodo.org/api/records/10960004"
DOI = "10.5281/zenodo.10960004"

# Small, diverse subset (sizes from the live API; one simpoint per workload).
SMALL_PRESET = (
    "649.fotonik3d_s-1B.champsimtrace.xz",  # smallest file; FP
    "654.roms_s-1021B.champsimtrace.xz",  # small FP
    "648.exchange2_s-1699B.champsimtrace.xz",  # small INT
)

ROOT = Path(__file__).resolve().parents[1]
TRACE_DIR = ROOT / "data" / "traces"
META_CSV = ROOT / "data" / "metadata" / "traces.csv"


def fetch_record() -> dict:
    req = urllib.request.Request(ZENODO_API, headers={"User-Agent": "anba-research/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_existing() -> dict[str, dict]:
    if not META_CSV.exists():
        return {}
    with META_CSV.open(newline="") as f:
        return {row["filename"]: row for row in csv.DictReader(f)}


def write_catalog(files: list[dict], existing: dict[str, dict]) -> None:
    META_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "filename",
        "workload",
        "bytes",
        "zenodo_md5",
        "sha256",
        "downloaded",
        "source_url",
        "doi",
        "record",
    ]
    rows = []
    for entry in sorted(files, key=lambda e: e.get("key", "")):
        name = entry["key"]
        workload = name.split("_s-")[0] if "_s-" in name else name.split(".")[0]
        prev = existing.get(name, {})
        checksum = entry.get("checksum") or ""
        md5 = checksum.split(":")[-1] if checksum else prev.get("zenodo_md5", "")
        url = entry.get("links", {}).get("self") or prev.get("source_url", "")
        downloaded = prev.get("downloaded", "no")
        sha256 = prev.get("sha256", "")
        local = TRACE_DIR / name
        if local.exists() and local.stat().st_size == int(entry.get("size") or 0):
            if not sha256:
                sha256 = sha256_file(local)
            downloaded = "yes"
        rows.append(
            {
                "filename": name,
                "workload": workload,
                "bytes": entry.get("size", ""),
                "zenodo_md5": md5,
                "sha256": sha256,
                "downloaded": downloaded,
                "source_url": url,
                "doi": DOI,
                "record": "10960004",
            }
        )
    with META_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote catalog {META_CSV} ({len(rows)} files)", file=sys.stderr)


def download_one(name: str, url: str, expected_size: int, expected_md5: str) -> str:
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    dest = TRACE_DIR / name
    if dest.exists() and dest.stat().st_size == expected_size:
        digest = sha256_file(dest)
        print(f"exists {name} sha256={digest}", file=sys.stderr)
        return digest
    print(f"downloading {name} ({expected_size} bytes)", file=sys.stderr)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "anba-research/0.1"})
    with urllib.request.urlopen(req, timeout=600) as resp, tmp.open("wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    if tmp.stat().st_size != expected_size:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"size mismatch for {name}: got {tmp.stat().st_size} expected {expected_size}")
    md5 = hashlib.md5()
    with tmp.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            md5.update(chunk)
    got_md5 = md5.hexdigest()
    if expected_md5 and got_md5 != expected_md5:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"md5 mismatch for {name}: got {got_md5} expected {expected_md5}")
    tmp.replace(dest)
    digest = sha256_file(dest)
    print(f"downloaded {name} sha256={digest}", file=sys.stderr)
    return digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("small", "none"), default="small")
    parser.add_argument("--files", nargs="*", default=None, help="explicit filenames from the catalog")
    parser.add_argument("--catalog-only", action="store_true")
    args = parser.parse_args()

    record = fetch_record()
    title = (record.get("metadata") or {}).get("title")
    files = record.get("files") or []
    if not files:
        print("Zenodo record returned no files", file=sys.stderr)
        return 1
    print(f"Zenodo 10960004 title={title!r} nfiles={len(files)}", file=sys.stderr)
    existing = load_existing()
    write_catalog(files, existing)

    if args.catalog_only:
        return 0

    wanted: list[str]
    if args.files:
        wanted = list(args.files)
    elif args.preset == "small":
        wanted = list(SMALL_PRESET)
    else:
        wanted = []

    by_name = {e["key"]: e for e in files}
    missing = [n for n in wanted if n not in by_name]
    if missing:
        print(f"requested files not in Zenodo catalog: {missing}", file=sys.stderr)
        return 1

    existing = load_existing()
    for name in wanted:
        entry = by_name[name]
        url = entry["links"]["self"]
        checksum = (entry.get("checksum") or "").split(":")[-1]
        digest = download_one(name, url, int(entry["size"]), checksum)
        existing[name] = existing.get(name, {})
        existing[name]["sha256"] = digest
        existing[name]["downloaded"] = "yes"
    write_catalog(files, existing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
