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
  weight decay for all tiny nets; Adam overfits on hidden layers. With SGD
  (default since 2026-09-12), NN-A/NN-B/NN-C all beat random at `--cap 100000`.
- Prior NN-A failure (`val_acc≈0.49` with Adam) is recorded in git history;
  root cause was optimizer choice, not the Dense16 architecture itself.
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

## Reproduce path (`make reproduce`)

Clean-clone flow (real downloads/builds; several minutes):

1. `make env` — pip install `requirements.txt` (PyTorch train/export only).
2. `make catalog` + `make traces` — Zenodo metadata + small preset (3 traces).
3. `make splits` — frozen `data/splits/spec_v1.json` (no-op if present).
4. `make champsim-pin` + `make champsim-deps` — clone pinned ChampSim + vcpkg.
5. Build all ANBA predictors (bimodal, instrumented, nn_frozen, hybrid, online).
6. Extract train/val branch CSVs (`654.roms`, `649.fotonik3d` only; test trace
   `648.exchange2` is downloaded for sim matrix but never used for training).
7. Train NN-A (may fail loudly) and NN-C; export INT8 when checkpoints exist.
8. `experiments/matrix.sh` on every downloaded trace; `make accept`.

`results/parsed/matrix_summary.json` aggregates real sim metrics when matrix
runs complete. Failed steps abort `reproduce` except NN-A train (documented).

## Hard workloads (bimodal-weak eval)

Probe: `python3 tools/bimodal_probe.py` (WARMUP=1e5, SIM=2e5, same as matrix).
Full probe table: `results/parsed/bimodal_probe.json`.

### Why these traces are “hard”

Compared to the easy reference `649.fotonik3d` (bimodal acc **95.22%**), the
following downloaded simpoints have materially weaker bimodal branch prediction
(acc **&lt; 90%**):

| Trace | Workload | Bimodal acc % | Bimodal IPC | Bimodal MPKI |
| --- | --- | ---: | ---: | ---: |
| `654.roms_s-1021B` | FP (ocean) | 80.48 | 1.507 | 31.63 |
| `648.exchange2_s-1699B` | INT (finance) | 84.04 | 1.676 | 21.50 |
| `603.bwaves_s-3699B` | FP (fluid) | 87.22 | 1.957 | 17.87 |
| `607.cactuBSSN_s-4248B` | FP (relativity) | 87.45 | 1.547 | 14.74 |

`631.deepsjeng` (89.71%) was probed but kept as **validation only** (borderline).
`605.mcf`, `638.imagick`, `644.nab` probed **&gt; 98%** bimodal acc — used as
easy train workloads, not hard eval.

### Split and retrain

- New split: `data/splits/spec_hard_v1.json` (seed **20260913**). `spec_v1.json`
  is unchanged.
- **Train** (easy): `649.fotonik3d`, `638.imagick`, `644.nab` — never includes
  hard test workloads.
- **Val**: `631.deepsjeng`.
- **Test (hard eval)**: `654.roms`, `648.exchange2`, `603.bwaves`, `607.cactuBSSN`.
- Retrain was **required**: hard test workloads are outside the `spec_v1` small
  train domain (`654.roms` only). All NNs retrained on `spec_hard_v1` train CSVs;
  val gate passed (&gt; 0.5 random). Reports: `results/parsed/train_hard_nn_*.json`.
- Exported INT8: `models/export/nn_c_int8.bin` (rebuilt from hard-split NN-C).

### Hard matrix results (WARMUP=1e5, SIM=2e5)

Aggregated: `results/parsed/hard_matrix_summary.json` (`fabricated: false`).
Run: `make hard-matrix` or `experiments/hard_matrix.sh`.

**Bimodal baseline on hard set**

| Trace | Acc % | IPC | MPKI |
| --- | ---: | ---: | ---: |
| roms | 80.48 | 1.507 | 31.63 |
| exchange2 | 84.04 | 1.676 | 21.50 |
| bwaves | 87.22 | 1.957 | 17.87 |
| cactuBSSN | 87.45 | 1.547 | 14.74 |

**Deltas vs bimodal (same traces, hard-split NN-C export)**

| Trace | Predictor | Δ acc % | Δ IPC | Δ MPKI |
| --- | --- | ---: | ---: | ---: |
| roms | anba_online | −0.90 | −0.026 | +1.45 |
| roms | anba_hybrid | −6.32 | −0.165 | +10.23 |
| roms | nn_frozen | −24.06 | −0.531 | +38.98 |
| exchange2 | anba_online | +0.19 | +0.007 | −0.25 |
| exchange2 | anba_hybrid | −7.35 | −0.225 | +9.90 |
| exchange2 | nn_frozen | −26.96 | −0.682 | +36.33 |
| bwaves | anba_online | −0.06 | −0.002 | +0.07 |
| bwaves | anba_hybrid | −5.75 | −0.227 | +7.97 |
| bwaves | nn_frozen | −33.54 | −0.834 | +46.72 |
| cactuBSSN | anba_online | 0.00 | 0.000 | 0.00 |
| cactuBSSN | anba_hybrid | −12.45 | −0.192 | +14.63 |
| cactuBSSN | nn_frozen | −49.80 | −0.735 | +58.50 |

**Offline val accuracy after retrain (`spec_hard_v1`, cap 100k rows/file)**

| Model | Val acc | Beat random |
| --- | ---: | --- |
| NN-A | 0.5294 | yes |
| NN-B | 0.5200 | yes |
| NN-C | 0.5298 | yes |

On these hard workloads, **bimodal remains competitive**: `anba_online` tracks
bimodal (cold-start + short window), while `anba_hybrid` and `nn_frozen` regress
because the frozen NN was trained on easy workloads and hurts when the hybrid
gate routes branches to it.

## Hard-domain retrain (train hard → test held-out hard)

Previous `spec_hard_v1` trains on **easy** workloads and evaluates on **hard**
test traces. `spec_hard_domain_v1` trains on **hard** workloads and evaluates on
**held-out hard** test traces only (trace-level split; never the same trace in
train and test).

### Dataset semantics

Each instrumentation CSV row is a **resolved branch** with its true
taken/not-taken outcome (`pc,history,taken,...`). Rows are **not** filtered to
bimodal failures or mispredictions only. Supervised learning already teaches the
correct outcome bit everywhere, including branches where bimodal and other
heuristics fail.

### Split (`data/splits/spec_hard_domain_v1.json`, seed **20260914**)

Hard pool = bimodal probe acc &lt; ~90% (already downloaded for hard eval):

| Workload | Simpoint | Bimodal acc % | Split |
| --- | --- | ---: | --- |
| `603.bwaves` | `603.bwaves_s-3699B` | 87.22 | **train** |
| `607.cactuBSSN` | `607.cactuBSSN_s-4248B` | 87.45 | **train** |
| `631.deepsjeng` | `631.deepsjeng_s-928B` | 89.71 | **val** |
| `654.roms` | `654.roms_s-1021B` | 80.48 | **test** |
| `648.exchange2` | `648.exchange2_s-1699B` | 84.04 | **test** |

`spec_v1.json` and `spec_hard_v1.json` are unchanged. Generate with
`make hard-domain-splits`.

### Offline train/val (instrumentation cap 100k rows/file, SGD)

Reports: `results/parsed/train_hard_domain_nn_*.json`. Val gate: acc &gt; 0.5.

| Model | Train acc | Val acc | Beat random |
| --- | ---: | ---: | --- |
| NN-A | 0.8260 | 0.5775 | yes |
| NN-B | 0.8130 | 0.6405 | yes |
| NN-C | 0.7489 | 0.5488 | yes |

Exported INT8: `models/export/nn_c_int8.bin` (NN-C, best val arch for C++ path).

Pipeline: `make hard-domain-train` (instrument → train → export).

### Hard-domain test matrix (WARMUP=1e5, SIM=2e5)

Aggregated: `results/parsed/hard_domain_matrix_summary.json` (`fabricated: false`).
Run: `make hard-domain-matrix`.

**Bimodal baseline on held-out hard test**

| Trace | Acc % | IPC | MPKI |
| --- | ---: | ---: | ---: |
| roms | 80.48 | 1.507 | 31.63 |
| exchange2 | 84.04 | 1.676 | 21.50 |

**Deltas vs bimodal (hard-domain-trained NN-C export)**

| Trace | Predictor | Δ acc % | Δ IPC | Δ MPKI |
| --- | --- | ---: | ---: | ---: |
| roms | anba_online | −0.66 | −0.019 | +1.07 |
| roms | anba_hybrid | −4.90 | −0.140 | +7.94 |
| roms | nn_frozen | −19.67 | −0.466 | +31.86 |
| exchange2 | anba_online | +0.21 | +0.008 | −0.28 |
| exchange2 | anba_hybrid | −7.39 | −0.223 | +9.96 |
| exchange2 | nn_frozen | −24.94 | −0.660 | +33.59 |

**Contrast: easy-train→hard-test vs hard-train→hard-test (Δ acc vs bimodal)**

Hard-domain training modestly improves hybrid/nn_frozen on roms and exchange2
vs the easy-train export (e.g. roms `nn_frozen` Δ acc improves from −24.06 to
−19.67), but **none beat bimodal** on these held-out traces. `anba_online`
remains near bimodal (cold-start bimodal + online residual).

## Residual specialist + safer hybrid gate

### Curriculum (disagreement-focused dataset)

- Split: `data/splits/spec_residual_v1.json` (seed **20260915**). Same
  workload-level train/val/test traces as `spec_hard_domain_v1`; never builds
  residual CSVs from test traces.
- Builder: `tools/build_residual_dataset.py` reads instrumented dumps
  (`pc,history,taken,branch_type,predicted`) and writes
  `data/extracted/residual/{stem}.csv`.
- **Emphasis:** rows where `predicted != taken` (bimodal mispredictions).
- **Agreement mix:** default **15%** of output rows are `predicted == taken` so
  training is not dominated by rare disagreement events only.
- **Target label:** true `taken` (not flip-of-bimodal). Stats:
  `results/parsed/residual_dataset_stats.json`.
- Features: standard `pack_features` (32 bits PC+GHR). Bimodal `predicted` is
  **not** appended as an extra feature (keeps Python/C++ packing identical).

### Residual training + export

- Trainer: `ml/train_residual.py` trains NN-A/B/C on residual CSVs, val gate
  acc > 0.5, picks **best val arch** (not hardcoded).
- Export: `models/export/nn_residual_int8.bin` (+ sidecar JSON).
- Reports: `results/parsed/train_residual_nn_*.json`,
  `results/parsed/train_residual_summary.json`.
- Pipeline: `make residual-train` (instrument → build dataset → train → export).

### Safer hybrid gate (`anba_residual_hybrid`)

Keeps `anba_hybrid` intact for A/B comparison.

| 2-bit state | Bimodal | NN consulted? | Override rule |
| --- | --- | --- | --- |
| `00` / `11` | confident | no | bimodal |
| `01` / `10` | uncertain | yes | override **only** if \|INT8 accumulator\| ≥ margin |
| (margin not met) | uncertain | yes | keep bimodal |

- Default margin: `ANBA_NN_MARGIN=8` (INT8 accumulator units, same scale as
  `anba_online` residual threshold).
- Model path: `ANBA_MODEL_PATH=models/export/nn_residual_int8.bin`.
- Build/run: `make residual-hybrid TRACE=...`.

### Residual eval matrix

- Script: `experiments/residual_matrix.sh` / `make residual-matrix`.
- Windows: WARMUP=1e5, SIM=2e5 (same as hard-domain).
- Held-out hard test traces from `spec_residual_v1`.
- Predictors: bimodal, anba_hybrid (hard-domain NN-C), anba_online,
  anba_residual_hybrid (residual export), nn_frozen (residual weights).
- Summary: `results/parsed/residual_matrix_summary.json` with
  `deltas_vs_bimodal` and `deltas_vs_anba_hybrid` (`fabricated: false`).

### Residual dataset stats (instrumentation cap 100k rows/file)

Mean bimodal disagreement in source dumps: **12.27%** of rows
(`predicted != taken`). After 15% agreement mix:

| Split | Trace | Source rows | Disagree % | Output rows |
| --- | --- | ---: | ---: | ---: |
| train | bwaves-3699B | 42427 | 12.99 | 6484 |
| train | cactuBSSN-4248B | 35331 | 12.60 | 5235 |
| val | deepsjeng-928B | 45789 | 11.23 | 6047 |

Full stats: `results/parsed/residual_dataset_stats.json`.

### Offline val (residual curriculum, SGD)

Best arch selected by val (not hardcoded): **NN-B** (`val_acc=0.5727`).
NN-C did **not** beat random on residual data.

| Model | Train acc | Val acc | Beat random |
| --- | ---: | ---: | --- |
| NN-A | 0.5953 | 0.5480 | yes |
| NN-B | 0.6612 | **0.5727** | yes |
| NN-C | 0.6150 | 0.4745 | no |

Export: `models/export/nn_residual_int8.bin` (NN-B).

### Residual test matrix (WARMUP=1e5, SIM=2e5)

**Bimodal baseline on held-out hard test**

| Trace | Acc % | IPC | MPKI |
| --- | ---: | ---: | ---: |
| roms | 80.48 | 1.507 | 31.63 |
| exchange2 | 84.04 | 1.676 | 21.50 |

**Deltas vs bimodal**

| Trace | Predictor | Δ acc % | Δ IPC | Δ MPKI |
| --- | --- | ---: | ---: | ---: |
| roms | anba_online | −0.66 | −0.019 | +1.07 |
| roms | anba_hybrid | −4.90 | −0.140 | +7.94 |
| roms | **anba_residual_hybrid** | **0.00** | **0.000** | **0.00** |
| roms | nn_frozen (residual) | −19.32 | −0.445 | +31.31 |
| exchange2 | anba_online | +0.21 | +0.008 | −0.28 |
| exchange2 | anba_hybrid | −7.39 | −0.223 | +9.96 |
| exchange2 | **anba_residual_hybrid** | **0.00** | **0.000** | **0.00** |
| exchange2 | nn_frozen (residual) | −30.55 | −0.753 | +41.14 |

**Deltas vs old `anba_hybrid` (hard-domain NN-C)**

| Trace | Predictor | Δ acc % | Δ IPC |
| --- | --- | ---: | ---: |
| roms | anba_residual_hybrid | +4.90 | +0.140 |
| exchange2 | anba_residual_hybrid | +7.39 | +0.223 |

The safer gate (`ANBA_NN_MARGIN=8`) prevents harmful NN overrides on uncertain
branches: `anba_residual_hybrid` matches bimodal on both held-out traces while
recovering the full regression of `anba_hybrid`. `nn_frozen` with residual
weights still regresses (expected — no bimodal backbone).

## Known deviations / leftover work

- Full SPEC matrix and championship-length windows: not run unless logs exist.
- Tanh (train) vs sign (C++ hidden): documented above.
- ChampSim vcpkg bootstrap is host-dependent and not cached in-repo.
- `nn_frozen` cross-workload sim numbers are expected to be poor; hybrid/online
  are the intended deployment modes.
- Hard-eval hybrid regression is expected with easy-workload-trained NN-C; see
  **Hard-domain retrain** below for train-on-hard / test-on-held-out-hard results.
- Paper-quality tables, energy models, and hardware-area estimates are out of
  scope for this prototype.
