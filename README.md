# Adaptive Neural Branch Assistance (ANBA)

ChampSim research prototype: a **confidence-gated hybrid** that uses a tiny
offline-trained neural branch predictor only when a conventional 2-bit
bimodal counter is uncertain (`01` / `10`). Strong conventional states
(`00` / `11`) keep the bimodal prediction.

This repository is a runnable prototype, not a published result. Every
reported IPC / MPKI / accuracy number must come from a real simulator run
or is marked `unavailable`. No fabricated metrics.

## Non-negotiables

- Same ChampSim CPU/memory config (`configs/cpu/base.json`) for every
  predictor. Only `branch_predictor` changes.
- Simulator neural inference is **native C++ INT8** (`predictor/common/anba_int8.h`).
  PyTorch is used for train/export only.
- Splits are **workload-level** (SPEC benchmark), frozen in
  `data/splits/spec_v1.json` with seed `20260912`. The test split is never
  used for training.
- Traces are official SPEC CPU 2017 ChampSim traces from
  [Zenodo 10960004](https://zenodo.org/records/10960004). They are not
  committed. SHA256 is recorded only after a real download.

## Layout

| Path | Role |
| --- | --- |
| `configs/` | Frozen CPU config |
| `data/metadata/` | Software pin + trace catalog |
| `data/splits/` | Frozen train/val/test |
| `champsim/` | Pin notes; clone is gitignored |
| `predictor/` | Instrumented, frozen NN, hybrid, online |
| `ml/` | PyTorch train + INT8 export |
| `tools/` | Download, parse, splits, acceptance |
| `experiments/` | Driver scripts |
| `results/` | Real logs + parsed JSON |
| `models/` | Checkpoints + INT8 export |
| `paper/` | Notes only (no invented tables) |
| `docs/IMPLEMENTATION_NOTES.md` | Deviations |

## Quick start

```bash
# 1. Catalog + small official traces (~100 MB compressed)
make catalog
make traces

# 2. Pin and build official ChampSim (vcpkg; several minutes)
make champsim

# 3. Baseline on one real trace (short instruction window)
make baseline

# 4. Freeze splits, extract features, train, export, hybrid
make splits
make extract
make train ARCH=nn_c
make export-int8 ARCH=nn_c
make champsim-build PREDICTOR=nn_frozen
make nn
make hybrid
make accept
```

`make reproduce` walks the implemented pipeline end-to-end on a clean clone
(catalog, small traces, vcpkg bootstrap, all predictors, train/export, matrix)
and writes `results/parsed/acceptance.json` plus `matrix_summary.json`.

```bash
make reproduce   # ~10–20 min first time (vcpkg + five ChampSim binaries)
make accept      # re-check A1–A11 from artifacts only
make matrix      # re-run sim matrix on downloaded traces
```

## Predictors

| Name | Behavior |
| --- | --- |
| `bimodal` / `gshare` / `hashed_perceptron` | ChampSim built-ins (baseline) |
| `instrumented` | Bimodal + CSV dump of PC, 16-bit GHR, outcome |
| `nn_frozen` | INT8 MLP/perceptron, no online update |
| `anba_hybrid` | 2-bit 00/11 conventional, 01/10 neural |
| `anba_online` | Cold-start with conventional, then hybrid + tiny residual update |

Set `ANBA_MODEL_PATH` to an exported `models/export/*_int8.bin`.

## Networks (train/export)

- **NN-A** `Dense16`: 32 → 16 → 1
- **NN-B** `Dense32-8`: 32 → 32 → 8 → 1
- **NN-C** perceptron-like: 32 → 1

Features (32 × ±1): 16 PC bits of `(ip >> 2)` and 16 global-history bits.
Acceptance gate for training: validation accuracy **must beat random (0.5)**
or the trainer exits non-zero.

## ChampSim pin

Recorded in `data/metadata/software_versions.json`. Clone is not vendored.

## Traces

`make traces` fetches the **small** preset (one simpoint each):

- `649.fotonik3d_s-1B.champsimtrace.xz`
- `654.roms_s-1021B.champsimtrace.xz`
- `648.exchange2_s-1699B.champsimtrace.xz`

The full 95-file catalog is listed in `data/metadata/traces.csv` from the
live Zenodo API. Do not invent additional traces.

Short default windows (`WARMUP=100000`, `SIM=200000`) are for a working
prototype, not a championship-length result. Override on the make command
line for longer runs.

## Findings

Full narrative (all phases, bugs, charts): **[paper/ANBA_SIDE_TRY.md](paper/ANBA_SIDE_TRY.md)**.  
Every headline number below is from committed `results/parsed/*` with `fabricated: false`.

### Headline (measured)

| Question | Answer |
| --- | --- |
| Gated **residual MLP** or **NN-T** vs **bimodal** on hard held-out (`roms`, `exchange2`) | **Tie** — **0.00 pp**, **0 IPC** |
| **Ungated** hybrid on uncertain branches | **~−5 to −7 pp** vs bimodal |
| Frozen specialists on **unseen** SPEC traces | **Lose** unless gate margin is high enough to never override (then tie) |
| **gshare-hybrid** (`anba_gshare_online`) vs **gshare** (online v2, 100k/500k) | **+0.09 pp** on `641.leela` only; **0.00** on gcc; **no IPC win** (≤0.002 vs gshare) |
| vs **hashed_perceptron** (same v2 matrix) | **hashed_perceptron ahead** by **+0.04 to +3.55 pp** on every v2 test trace |
| **Bimodal-only** baselines | **Too weak** — gshare leads bimodal by up to **~11.5 pp** on hard traces in repeating-task runs |

### Online v2 — Δ branch accuracy vs **gshare** (pp, epoch 5)

| Trace | hashed_perceptron | gshare-hybrid | bimodal (≈ vs gshare) |
| --- | ---: | ---: | ---: |
| `641.leela` | +3.55 | **+0.09** | −0.88 |
| `602.gcc` | +0.04 | 0.00 | −5.21 |
| `657.xz` | +2.08 | 0.00 | −1.04 |
| `600.perlbench` | +0.54 | −0.02 | −0.25 |
| `654.roms` | +2.37 | −1.98 | −15.80 |
| `648.exchange2` | +2.61 | −1.00 | −11.78 |

Source: `results/parsed/online_repeating_v2_summary.json`.

### Lessons

1. **Gate** uncertain branches; ungated neural override is unsafe.  
2. Compare against **gshare** / **hashed_perceptron**, not bimodal alone.  
3. **NN-T** did not beat the residual MLP on test; both **tie bimodal** when gated.  
4. **Vote** aggregate had a **combined-vs-model_pred** bug (fixed); design still fails on gcc.  
5. Next experiment (not run): online corrector on **hashed_perceptron** — expect **tenths of a pp** at best if anything.

### Reproduce main result bundles

```bash
make residual-matrix                    # gated residual vs bimodal (hard test)
bash experiments/transformer_matrix.sh # NN-T + gate
bash experiments/unseen_programs_matrix.sh
bash experiments/online_repeating_v2.sh  # gshare / hashed_perc / hybrid
python3 -m unittest tests/test_online_vote_aggregate.py
```

Bootstrap smoke matrix (3 traces, early prototype): `make matrix` → `results/parsed/matrix_summary.json`.

## Acceptance (A1–A11)

See `docs/IMPLEMENTATION_NOTES.md` and `make accept`.

## License

MIT for ANBA scaffolding. ChampSim remains under its own license (Apache-2.0).
SPEC traces remain under the Zenodo record terms / SPEC license constraints.

## Research note

Consolidated findings: [paper/ANBA_SIDE_TRY.md](paper/ANBA_SIDE_TRY.md) (figures under `paper/figures/`).
