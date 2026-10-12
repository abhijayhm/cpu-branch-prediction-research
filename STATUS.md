# NN-T (transformer) branch — status checkpoint

**Last updated:** 2026-10-12 (UTC)  
**Branch:** `cursor/nn-t-transformer-variant-77e6`

## Done

- **NN-T training pipeline** (`ml/nn_t_model.py`, `ml/transformer_data.py`, `ml/train_residual_transformer.py`, `ml/export_transformer.py`): causal transformer over last 64 branch events (11-bit PC hash + taken / current-PC tokens), 2 layers, d_model=48, 4 heads; residual curriculum on ordered instrumented traces; val margin tune (selected **logit margin 8.0**).
- **Offline train (val only, no test leakage):** `results/parsed/train_residual_nn_t.json` — val acc **59.7%** (beats random); val gated sim acc **89.5%**; override rate **39.4%**, override precision **57.9%** on deepsjeng val (offline replay).
- **C++ predictors:** `nn_transformer_frozen`, `anba_transformer_hybrid` + `predictor/common/anba_transformer.h` (float export v2). Existing NN-A/B/C and residual paths unchanged.
- **Held-out hard test (warmup 100k / sim 200k)** — `results/parsed/transformer_matrix_summary.json` (`fabricated: false`):

| Trace | Bimodal acc / IPC | Residual+gate Δ | **NN-T+gate** acc / IPC | Δ vs bimodal |
|---|---:|---:|---:|---:|
| `654.roms` | 80.48% / 1.507 | 0 / 0 | **80.48% / 1.507** | **0 pp / 0 IPC** |
| `648.exchange2` | 84.04% / 1.676 | 0 / 0 | **84.04% / 1.676** | **0 pp / 0 IPC** |

- **Timing verification:** `anba_transformer_hybrid` full window on roms ≈ **12 s**. `nn_transformer_frozen` full window **>4 min** (aborted); reduced smoke **10k/20k** ≈ **34 s** per trace (metrics not comparable to 100k/200k bimodal table).
- **Makefile targets:** `transformer-train`, `transformer-matrix`; smoke script `experiments/transformer_matrix_smoke.sh`.
- **Paper draft section:** appended to `paper/ANBA_SIDE_TRY.md`; chart `paper/figures/nn_t_hard_accuracy_delta.png` (when chart script run).

## Remaining / not finished

- **Full-window `nn_transformer_frozen` alone** on both test traces (too slow at current C++ forward; needs KV-cache or slimmer export).
- **Override rate/precision on test** from ChampSim (destructor `ANBA_GATE_STATS` not appearing in captured logs; offline val stats only).
- **Optional seq_len=128** ablation (not run).
- **C++ vs PyTorch logit parity test** (`tools/verify_transformer_export.py` stub only).
- **Re-run `make transformer-matrix` end-to-end** after perf work (partial run completed hybrid + residual; frozen skipped).

## Verdict (measured)

NN-T with the same confidence gate **ties bimodal** on both held-out hard short windows (same outcome as residual MLP+gate). Ungated / frozen-only NN-T at full sim cost is **not** evaluated at production windows yet; offline val shows the specialist can beat random but does not prove an IPC win.
