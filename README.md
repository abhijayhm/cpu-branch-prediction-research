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
make train ARCH=nn_a
make export-int8 ARCH=nn_a
make champsim-build PREDICTOR=nn_frozen
make nn
make hybrid
make accept
```

`make reproduce` walks the implemented pipeline and writes
`results/parsed/acceptance.json`.

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

## Acceptance (A1–A11)

See `docs/IMPLEMENTATION_NOTES.md` and `make accept`.

## License

MIT for ANBA scaffolding. ChampSim remains under its own license (Apache-2.0).
SPEC traces remain under the Zenodo record terms / SPEC license constraints.
