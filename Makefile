# ANBA research prototype. Real artifacts only; no fabricated metrics.
SHELL := /bin/bash
ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
CS := $(ROOT)/champsim/ChampSim
PIN := $(ROOT)/data/metadata/software_versions.json
PREDICTOR ?= bimodal
WARMUP ?= 100000
SIM ?= 200000
TRACE ?= $(ROOT)/data/traces/649.fotonik3d_s-1B.champsimtrace.xz
MODEL ?= $(ROOT)/models/export/nn_c_int8.bin
ARCH ?= nn_c
PY ?= python3
NPROC ?= 2
# Default /usr/bin/c++ may be clang without libstdc++ symlink; use g++ for ChampSim/vcpkg.
export CC := gcc
export CXX := g++

.PHONY: help scaffold env champsim-pin champsim-deps champsim-build champsim \
	traces catalog splits hard-splits baseline parse instrument extract train export-int8 \
	nn hybrid online smoke matrix hard-matrix reproduce accept clean-results

help:
	@echo "ANBA targets:"
	@echo "  make catalog          Zenodo 10960004 metadata -> traces.csv"
	@echo "  make traces           download SMALL official traces"
	@echo "  make champsim-pin     clone ChampSim at pinned commit"
	@echo "  make champsim-deps    vcpkg bootstrap + install"
	@echo "  make champsim-build PREDICTOR=bimodal"
	@echo "  make baseline         run bimodal on TRACE (real sim)"
	@echo "  make splits           freeze data/splits/spec_v1.json"
	@echo "  make extract          dump (pc,history,taken) from TRACE"
	@echo "  make train ARCH=nn_a  train on frozen train split only"
	@echo "  make export-int8      write models/export/*_int8.bin"
	@echo "  make matrix           run predictors on all downloaded traces"
	@echo "  make hard-splits      freeze data/splits/spec_hard_v1.json"
	@echo "  make hard-matrix      run predictors on hard-eval traces"
	@echo "  make reproduce        run the implemented pipeline"
	@echo "  make accept           print A1-A11 from artifacts"

scaffold:
	@test -f $(ROOT)/configs/cpu/base.json
	@test -f $(ROOT)/data/metadata/software_versions.json
	@echo "scaffold ok"

env:
	$(PY) -m pip install --user --break-system-packages -r $(ROOT)/requirements.txt

catalog:
	$(PY) $(ROOT)/tools/download_traces.py --catalog-only

traces:
	$(PY) $(ROOT)/tools/download_traces.py --preset small

champsim-pin:
	bash $(ROOT)/tools/pin_champsim.sh

champsim-deps: champsim-pin
	cd $(CS) && ./vcpkg/bootstrap-vcpkg.sh
	cd $(CS) && ./vcpkg/vcpkg install

champsim-build: champsim-pin
	bash $(ROOT)/tools/install_predictors.sh
	$(PY) $(ROOT)/tools/gen_champsim_config.py --predictor $(PREDICTOR) \
		--out $(ROOT)/champsim/generated/$(PREDICTOR).json
	cd $(CS) && ./config.sh $(ROOT)/champsim/generated/$(PREDICTOR).json
	$(MAKE) -C $(CS) -j$(NPROC)

champsim: champsim-deps
	$(MAKE) champsim-build PREDICTOR=bimodal

BIN = $(CS)/bin/champsim_$(PREDICTOR)
ifeq ($(wildcard $(BIN)),)
BIN = $(CS)/bin/champsim
endif

baseline:
	@test -x $(CS)/bin/champsim_bimodal || test -x $(CS)/bin/champsim || \
		{ echo "build bimodal first: make champsim-build PREDICTOR=bimodal"; exit 1; }
	$(PY) $(ROOT)/tools/run_sim.py \
		--bin $$(test -x $(CS)/bin/champsim_bimodal && echo $(CS)/bin/champsim_bimodal || echo $(CS)/bin/champsim) \
		--trace $(TRACE) --predictor bimodal --warmup $(WARMUP) --sim $(SIM)

parse:
	$(PY) $(ROOT)/tools/parse_champsim.py $(firstword $(wildcard $(ROOT)/results/raw/*.log))

splits:
	$(PY) $(ROOT)/tools/make_splits.py

hard-splits:
	$(PY) $(ROOT)/tools/make_hard_splits.py

extract:
	$(PY) $(ROOT)/tools/extract_branches.py --trace $(TRACE)

instrument:
	@test -x $(CS)/bin/champsim_instrumented
	mkdir -p $(ROOT)/data/extracted
	ANBA_INSTRUMENT_OUT=$(ROOT)/data/extracted/instrumented.csv \
	ANBA_INSTRUMENT_CAP=200000 \
	$(PY) $(ROOT)/tools/run_sim.py \
		--bin $(CS)/bin/champsim_instrumented \
		--trace $(TRACE) --predictor instrumented --warmup $(WARMUP) --sim $(SIM)
	$(PY) $(ROOT)/tools/verify_instrumentation.py --dump $(ROOT)/data/extracted/instrumented.csv

train:
	$(PY) $(ROOT)/ml/train.py --arch $(ARCH)

export-int8:
	$(PY) $(ROOT)/ml/export_int8.py --arch $(ARCH)

nn:
	$(MAKE) champsim-build PREDICTOR=nn_frozen
	ANBA_MODEL_PATH=$(MODEL) $(PY) $(ROOT)/tools/run_sim.py \
		--bin $(CS)/bin/champsim_nn_frozen \
		--trace $(TRACE) --predictor nn_frozen --warmup $(WARMUP) --sim $(SIM) \
		--extra-env ANBA_MODEL_PATH=$(MODEL)

hybrid:
	$(MAKE) champsim-build PREDICTOR=anba_hybrid
	ANBA_MODEL_PATH=$(MODEL) $(PY) $(ROOT)/tools/run_sim.py \
		--bin $(CS)/bin/champsim_anba_hybrid \
		--trace $(TRACE) --predictor anba_hybrid --warmup $(WARMUP) --sim $(SIM) \
		--extra-env ANBA_MODEL_PATH=$(MODEL)

online:
	$(MAKE) champsim-build PREDICTOR=anba_online
	ANBA_MODEL_PATH=$(MODEL) $(PY) $(ROOT)/tools/run_sim.py \
		--bin $(CS)/bin/champsim_anba_online \
		--trace $(TRACE) --predictor anba_online --warmup $(WARMUP) --sim $(SIM) \
		--extra-env ANBA_MODEL_PATH=$(MODEL)

smoke: baseline

matrix:
	bash $(ROOT)/experiments/matrix.sh

hard-matrix:
	bash $(ROOT)/experiments/hard_matrix.sh

reproduce: env catalog traces splits champsim-pin
	@echo "=== reproduce: env/catalog/traces/splits/pin ==="
	@if [[ ! -d $(CS)/.git ]]; then echo "ChampSim pin failed"; exit 1; fi
	$(MAKE) champsim-deps || { echo "vcpkg bootstrap failed; see champsim/ChampSim/vcpkg"; exit 1; }
	$(MAKE) champsim-build PREDICTOR=bimodal
	$(MAKE) champsim-build PREDICTOR=instrumented
	$(MAKE) champsim-build PREDICTOR=nn_frozen
	$(MAKE) champsim-build PREDICTOR=anba_hybrid
	$(MAKE) champsim-build PREDICTOR=anba_online
	@for tr in $(ROOT)/data/traces/654.roms_s-1021B.champsimtrace.xz \
		$(ROOT)/data/traces/649.fotonik3d_s-1B.champsimtrace.xz; do \
		test -f "$$tr" && $(MAKE) extract TRACE="$$tr" || echo "skip extract $$tr"; \
	done
	@test -f $(ROOT)/data/traces/649.fotonik3d_s-1B.champsimtrace.xz && \
		$(MAKE) instrument TRACE=$(ROOT)/data/traces/649.fotonik3d_s-1B.champsimtrace.xz
	-$(PY) $(ROOT)/ml/train.py --arch nn_a || echo "nn_a train did not beat random (documented)"
	-$(PY) $(ROOT)/ml/train.py --arch nn_c
	-test -f $(ROOT)/models/nn_c.pt && $(MAKE) export-int8 ARCH=nn_c
	-test -f $(ROOT)/models/nn_a.pt && $(MAKE) export-int8 ARCH=nn_a || true
	-bash $(ROOT)/experiments/matrix.sh
	$(PY) $(ROOT)/tools/acceptance.py

accept:
	$(PY) $(ROOT)/tools/acceptance.py

clean-results:
	rm -f $(ROOT)/results/raw/*.log
