# ChampSim traces (not committed)

Traces are official SPEC CPU 2017 ChampSim traces from
[Zenodo record 10960004](https://zenodo.org/records/10960004)
(`doi:10.5281/zenodo.10960004`).

This directory is gitignored. Fetch a small subset with:

```
make traces
```

or:

```
python3 tools/download_traces.py --preset small
```

Catalog + checksums live in `data/metadata/traces.csv`. SHA256 is filled only
after a real download. Do not invent traces or checksums.
