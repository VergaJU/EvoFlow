# EvoFlow Agent Guidelines

This document provides context and instructions for AI agents working on the EvoFlow codebase.

## 1. Project Overview
EvoFlow is a Python-based tool for biological network analysis using MaxFlow min-cut and genetic algorithms (GA). It combines `networkx` (graph algorithms) with `deap` (evolutionary computation) and `scanpy` (single-cell data).

## 2. Directory Structure
- `EvoFlow/`: Core package
  - `cli.py`: Entry point, argparse-based CLI
  - `core.py`: GA optimization (DEAP) and MaxFlow (NetworkX)
  - `biology.py`: Network construction, expression normalization
  - `visualization.py`: Plotting utilities
  - `utils.py`: I/O helpers
- `tests/`: Unit tests (pytest)

## 3. Commands

### Run Application
```bash
python -m EvoFlow.cli \
  --network <path> \
  --annotation <path> \
  --file <path_to_h5ad> \
  --regulons <path> \
  --receptors "geneA|geneB" \
  --mediators "geneC,geneD" \
  --ligands <path_csv> \
  --label <obs_column> \
  --celltype <cell_type> \
  --n_jobs <int>
```

### CI/Lint (exact order as CI.yml)
```bash
ruff format --check .
ruff check .
pytest
```

### Pre-commit Hooks
```bash
pip install pre-commit
pre-commit install
```

## 4. Code Style

- **Quotes**: Double quotes `"` for consistency
- **Indentation**: 4 spaces
- **Graph naming**: `G` (main graph), `SG` (subgraph)
- **Node attributes**: `node_type`, `percolation`
- **Edge attributes**: `weight`, `edge_type`, `capacity`
- **Type hints**: Add to new functions and refactored code

## 5. Architecture Notes

- **GA**: Uses DEAP with global `creator` registration. `GAOptimizer` in `core.py` handles this.
- **Evaluation**: Designed for `multiprocessing` - use top-level functions (picklable).
- **Caching**: Evaluation results are cached in the main process via `SmartCacheMap` using `tuple(ind)` keys.
  Cache hits avoid IPC entirely. The cache is local to each `run()` call and lives for its duration.
- **Partial application**: `functools.partial` is used to bind evaluation parameters (graph, expression data, etc.) into a single top-level callable, keeping the signature picklable for multiprocessing.
- **Parallelism**: `GAOptimizer` accepts `n_jobs` to control the number of parallel worker processes. `_get_n_jobs()` validates the value (raises `ValueError` if < 1, clamps to `cpu_count()` with a warning if too large).
- **Graph preprocessing**: Adds "SuperSource" and "SuperSink" nodes for flow computation.
- **Active subgraph**: Only paths connecting ligands to targets are optimized.
- **Output files**: Prefixed with input filename (via `utils.py`)

## 6. Gotchas

- **Safety**: Never remove `if __name__=='__main__':` blocks - required for multiprocessing.
- **Path handling**: Always use absolute paths or resolve relative paths correctly.
- **Error handling**: Use specific `try/except` blocks, not bare `except:`.
- **Missing keys**: Use `.get()` or explicit checks for DataFrame/dict keys.
- **Performance**: Graph algorithms on large networks are slow - profile before optimizing.