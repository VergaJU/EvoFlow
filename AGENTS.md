# EvoFlow Agent Guidelines

Context and instructions for AI agents working on the EvoFlow codebase.

## 1. Project Overview

EvoFlow finds optimal node knock-outs in gene regulatory networks by combining MaxFlow min-cut analysis with a genetic algorithm (GA). Stack: `networkx` (graph structure), `scipy.sparse.csgraph` (flow computation), `deap` (GA), `scanpy`/`anndata` (single-cell data), `pygam` (parameter estimation).

## 2. Directory Structure

- `EvoFlow/`: Core package
  - `cli.py`: Entry point (`main()` argparse CLI + `run_evoflow()` Python API)
  - `core.py`: GA optimization (DEAP), flow computation, `SmartCacheMap`, GAM parameter guess
  - `biology.py`: `BioNetworkPreparer` — network construction, percolation from expression (`normalize_expression`, `EXPRESSION_STATS`/`EXPRESSION_NORMS` constants imported by the CLI)
  - `preparer.py`: `prepare_flow_topology()` — adds SuperSource/SuperSink/dummy nodes; `lock_targets` bitmask (0–3) locks original sources/sinks out of GA removal
  - `features_extraction.py`: Graph feature extraction (`NetworkFeatures` class)
  - `gam_preprocessing.py`: Data preprocessing pipeline for GAM training (`Preprocessing` class)
  - `model.py`: GAM wrapper (training, prediction, `rank()`)
  - `train.py`: CLI to retrain the GAM model from collected data
  - `visualization.py`: Plotting utilities
  - `utils.py`: I/O helpers (JSON/YAML/pickle/TSV savers)
  - `models/`: Pre-trained GAM (`model.pkl`), `config.yaml`, `training_data.csv`, `model_scores.csv`
- `tests/`: Unit tests (pytest), shared fixtures in `tests/conftest.py`
- `data/`: Example input files (network, annotation, regulons, ligands, h5ad) — usable for smoke runs

## 3. Commands

### Environment

This checkout has a ready gitignored `venv/` with all dev deps installed. Prefer it — its `ruff` is **0.17.5**, the exact version CI pins:

```bash
venv/bin/python -m pytest ...     # run tests
venv/bin/ruff format --check .    # matches CI ruff
venv/bin/evoflow --help           # installed console script
```

Fresh setup: `pip install -r requirements.txt -r requirements-dev.txt`

### Verification (exact CI order, `.github/workflows/ci.yml`)

```bash
ruff format --check .
ruff check .
pytest
```

- CI pins `ruff==0.17.5` — treat it as source of truth; `.pre-commit-config.yaml` pins the same version, so local, pre-commit and CI results agree.
- CI runs on pushes/PRs to `master` only (also pushes to `fix_expression_matrix`).

### Tests

- Full suite: `venv/bin/python -m pytest` — 153 tests, ~7s, fully offline (no services/data files needed).
- Single test: `venv/bin/python -m pytest tests/test_core.py::test_name` (list: add `--collect-only -q`).
- Fixtures (mock graphs, AnnData, network frames) live in `tests/conftest.py`.

### Run Application

```bash
python -m EvoFlow.cli \
  --network <path> \
  --annotation <path> \
  --file <path_to_h5ad_or_bulk_csv> \
  --regulons <path> \
  --receptors "geneA|geneB" \
  --mediators "geneC,geneD" \
  --ligands <path_csv> \
  --label <obs_column> \
  --celltype <cell_type> \
  --expression_stat <nonzero_ratio|mean|median|p90|max> \
  --expression_norm <none|cpm|log1p|minmax> \
  --expression_transpose \
  --n_jobs <int>
```

**CLI semantics (verified by `tests/test_cli.py`):**
- `--pop_size`/`--early_stop` default to unset: both given → used directly (GAM guessing auto-disabled by `run_evoflow`); both omitted → GAM guesses; half-specified → argparse error. Same rule applies in the Python API (`population_size`/`early_stop_gen`).
- Boolean flags are `argparse.BooleanOptionalAction` (`--flag` / `--no-flag`): `--repair`, `--delta_penalty`, `--norm_hypervolume`, `--dynamic_rates`, `--guess_parameters`, `--expression_transpose` — passing a string value (e.g. `--repair False`) is an error, unlike the old `type=bool`.
- `--label`/`--celltype` are optional (h5ad cell filtering only); half-specified → argparse error (CLI) / `ValueError` (API). Given with bulk csv/tsv expression → ignored with a printed notice.
- `--expression_stat`/`--expression_norm` choices come from `EXPRESSION_STATS`/`EXPRESSION_NORMS` in `biology.py` — edit those tuples (not the argparse definitions) to add options. `--expression_transpose` orients genes-as-rows bulk files; rejected for h5ad.
- `cli.py` wiring is guarded by `tests/test_cli.py`, including end-to-end smoke runs (h5ad and bulk csv) — keep tests in sync when changing `run_evoflow()`/`main()` signatures (this drift caused the 2026-10 breakage).

## 4. Code Style

Ruff enforces formatting (double quotes, `target-version = "py310"`) via `pyproject.toml`. Repo-specific conventions:

- **Graph naming**: `G` (main graph), `SG` (active subgraph)
- **Node attributes**: `node_type`, `percolation`
- **Edge attributes**: `weight`, `edge_type`, `capacity`
- **Type hints**: Add to new functions and refactored code

## 5. Architecture Notes

- **GA**: DEAP with global `creator` registration, managed by `GAOptimizer` in `core.py`.
- **Flow computation**: Graph is converted once to an int32 CSR matrix (`nx.to_scipy_sparse_array`, capacities pre-scaled by `scale=1000` via `build_scaled_capacities`); workers call `scipy.sparse.csgraph.maximum_flow` on the CSR — not NetworkX max-flow. NetworkX is used only for graph structure and active-subgraph extraction.
- **Evaluation**: Must be picklable — top-level functions bound with `functools.partial`; workers receive the CSR via `multiprocessing.Pool(initializer=_init_worker, initargs=...)`. Never rebuild the graph inside workers.
- **Caching**: `SmartCacheMap` caches evaluations in the main process keyed by `tuple(ind)`; cache hits skip IPC entirely. Cache is local to one `run()` call.
- **Parallelism**: `GAOptimizer(n_jobs=...)`; `_get_n_jobs()` raises `ValueError` if `< 1`, clamps to `cpu_count()` with a warning if too large. `use_multiprocessing=False` runs sequentially.
- **Graph preprocessing**: `preparer.prepare_flow_topology()` adds "SuperSource"/"SuperSink" and dummy nodes; `lock_targets` bitmask (0–3) controls which original source/sink nodes are exempt from removal.
- **Percolation from expression**: `biology.compute_percolation(stat, norm)` reduces the samples×genes matrix to one value per annotation gene and scales by its max (values in (0,1], feeding `weight * (perc_u + perc_v) / 2` capacities — never let percolation grow unbounded, `build_scaled_capacities` would overflow int32). `nonzero_ratio` (default) counts expressing samples on the raw matrix and deliberately ignores `norm` (min-max would shift nonzero entries to 0); other stats (`mean`/`median`/`p90`/`max`) apply `norm` first (`cpm` = per-sample row scaling, `minmax` = per-gene across samples, constant gene → 0). Genes-as-rows bulk files are oriented by `transpose_expression()` in `load_data` (first non-numeric column becomes gene IDs, then `.T`); zero gene overlap with the annotation prints a transpose hint. Node attr name `percolation` is consumed by capacity updates and plots — do not rename.
- **Active subgraph**: Only paths connecting ligands to targets are optimized (`extract_active_subgraph`).
- **Removable nodes**: `set(SG.nodes) - set(sources) - set(sinks)`; node order in `SG` defines the CSR index space.
- **Outputs**: Written to the **current working directory**. Prefix = h5ad basename truncated at `"tmp"` (cli.py:83-85). Visualization/network files concatenate the prefix directly (no separator — real prefixes end with `_`), while `utils.py` savers add explicit `_` (e.g. `expr_ga_results.json` vs `exprnetwork.graphml`). The GA run log is `ga_run_<epoch>.jsonl` in CWD (covered by `.gitignore` `ga*.jsonl`).
- **GAM parameter estimation**: `GAOptimizer.guess_parameters_gam()` extracts 4 features (`log_search_space`, degree exponent `alpha`, `density`, `clustering_coef`) via `NetworkFeatures` → `pd.DataFrame` → `GAM.rank()` grid search → predicted `(population_size, patience)` pair.
- **GAM config**: Feature terms/constraints defined in `models/config.yaml`, read via `importlib.resources` — requires `EvoFlow/models/__init__.py`. `models/` is declared as package data in `pyproject.toml`; keep `*.pkl`/`*.yaml` when touching packaging.
- **Model file**: `models/model.pkl` stores `{"model": <LinearGAM>, "scores": <cv_results_df>}`; `load_model()` unwraps the dict for backward compatibility.
- **Retraining**: `venv/bin/python -m EvoFlow.train` — 10-fold CV (`KFold(n_splits=10, shuffle=True, random_state=42)`) with per-fold `gam.gridsearch`, then final fit; writes `model_scores.csv`.

## 6. Gotchas

- **Safety**: Never remove `if __name__=='__main__':` blocks — required for multiprocessing (spawn safety).
- **Multiprocessing picklability**: Evaluation functions and bound args must stay importable from module top level; closures/lambdas break the pool.
- **Guess-vs-specify**: `GAOptimizer.run()` refuses `population_size`/`early_stop_gen` set together with `guess_parameters=True`, and also refuses `guess_parameters=False` with either unset. `run_evoflow()` auto-disables guessing when values are provided and rejects half-specified pairs; set both explicitly or let the GAM guess both.
- **Whole-repo lint**: `ruff format --check .` covers untracked, non-gitignored files — local scratch files at repo root (e.g. `debug_evaluate.py`) fail the check even though CI only sees committed files.
- **Performance**: Flow/GA evaluation on large networks is slow — profile before optimizing; `SmartCacheMap` and CSR conversion exist for this reason.
