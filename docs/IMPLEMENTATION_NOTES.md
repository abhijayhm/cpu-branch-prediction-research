# Implementation notes and deviations

This file records what the prototype actually does versus a full paper-scale
study. It is not a results table.

## Pin

- ChampSim: `https://github.com/ChampSim/ChampSim` commit
  `410cee62d9c429c043ea611325ce00e6c303be2a` (master as of 2026-09-04).
- Branch-predictor hook used by that commit:
  `predict_branch(champsim::address)` and
  `last_branch_result(ip, target, taken, type)`.
- ChampSim is cloned at build time, not vendored.

## Traces

- Source is Zenodo record 10960004 (SPEC CPU 2017 ChampSim traces).
- Catalog is written from the live API. SHA256 is filled only after download.
- The small preset is three real files (~97 MB compressed). Other catalog
  entries are metadata-only until explicitly downloaded.
- Huge traces are gitignored.

## CPU config

- `configs/cpu/base.json` is the ChampSim default single-core JSON from the
  pinned commit (fields copied from `champsim_config.json`).
- Generated configs change only `ooo_cpu[0].branch_predictor` and
  `executable_name`.

## Splits

- `tools/make_splits.py` freezes `data/splits/spec_v1.json` with seed
  `20260912`.
- Unit is the SPEC **workload** (e.g. `648.exchange2`), not an instruction
  cut. All simpoints of a workload stay together.
- Training code refuses to load test-split CSVs.

## Instrumentation

- `predictor/instrumented` dumps `pc,history,taken,branch_type,predicted`.
- History is a 16-bit global shift register.
- `tools/verify_instrumentation.py` checks the GHR update rule on a real dump.
- `tools/extract_branches.py` reads ChampSim `input_instr` records (64 bytes,
  `inc/trace_instruction.h`). If that layout is wrong for a given file, the
  extractor will produce garbage — compare against the instrumentation dump
  before training.

## Neural path

- Train with PyTorch (`ml/train.py`). Hidden train activations are `Tanh`.
- C++ INT8 inference uses **sign** activations on hidden layers and a
  threshold of 0 on the last accumulator. This is an approximation of Tanh
  and is called out in the export metadata.
- NN-C (perceptron-like) has no hidden nonlinearity, so train and C++ match
  after INT8 quantization (up to scale rounding).
- Training gate: validation accuracy must be > 0.5 (coin-flip random).
  Majority-class accuracy is reported separately and is **not** the gate.
- Cross-workload val (train `654.roms`, val `649.fotonik3d`) needs SGD +
  weight decay; Adam overfits. NN-C and NN-B pass with `--cap 100000`;
  NN-A (Dense16) did not beat random in our runs.
- Default export/sim model: `models/export/nn_c_int8.bin` (perceptron).

## Hybrid / online

- Hybrid: 2-bit bimodal `00`/`11` → conventional; `01`/`10` → neural.
- Online: conventional-only for the first `ANBA_COLDSTART_N` branches
  (default 2048), then the same gate, plus a 32-wide INT8 residual
  perceptron updated with a perceptron learning rule.

## Simulation length

- Default `WARMUP=1e5`, `SIM=2e5` retired instructions. These are smoke /
  prototype windows, not 200M/500M championship runs.
- Longer runs are supported (`make baseline WARMUP=200000000 SIM=500000000`)
  but are not claimed here unless a parsed log exists.

## Acceptance A1–A11

| ID | Meaning |
| --- | --- |
| A1 | Scaffold (configs, Makefile, requirements) |
| A2 | environment.yml, Dockerfile, LICENSE |
| A3 | ChampSim pin matches `software_versions.json` |
| A4 | Real traces cataloged; ≥1 downloaded with SHA256 |
| A5 | ChampSim binary built |
| A6 | ≥1 baseline on ≥1 real trace; IPC/cycles/MPKI/accuracy parsed |
| A7 | Instrumentation verified (PC, GHR, outcome) |
| A8 | Frozen workload-level splits, seed 20260912 |
| A9 | Tiny NN beats random on validation |
| A10 | INT8 export + C++ `nn_frozen` binary |
| A11 | Hybrid binary + this notes file |

`make accept` reads artifacts only. Failed or un-run items stay `pass: false`
with an explicit detail string. No placeholder numbers.

## Build note (g++)

Cloud images may default `/usr/bin/c++` to clang without `libstdc++`.
The Makefile exports `CC=gcc` and `CXX=g++` for ChampSim/vcpkg builds.

## Known deviations / leftover work

- Full SPEC matrix and championship-length windows: not run unless logs exist.
- Tanh (train) vs sign (C++ hidden): documented above.
- ChampSim vcpkg bootstrap is host-dependent and not cached in-repo.
- Paper-quality tables, energy models, and hardware-area estimates are out of
  scope for this prototype.
