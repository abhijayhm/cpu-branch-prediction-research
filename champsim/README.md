# ChampSim pin

Official simulator: https://github.com/ChampSim/ChampSim

Pinned commit is recorded in `data/metadata/software_versions.json`.

```
make champsim-pin    # clone / fetch and checkout the pin
make champsim-deps   # vcpkg bootstrap + install
make champsim-build PREDICTOR=bimodal
```

The clone lives in `champsim/ChampSim/` and is gitignored. ANBA predictors
under `predictor/` are copied into `champsim/ChampSim/branch/` at build time.

All ANBA binaries share `configs/cpu/base.json`. Only `branch_predictor` changes.
