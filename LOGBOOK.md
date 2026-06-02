# EvoFlow Logbook

## Session: Add n_jobs Multiprocessing Support

**Date:** 2026-06-02
**Branch:** `feature_multiprocessing_njobs_core_001`
**Commit:** `12fb8ff`

### Summary
Added `n_jobs` parameter to `GAOptimizer` to control the number of parallel worker processes for multiprocessing, with proper bounds validation.

### Changes Made

**`EvoFlow/core.py`:**
- Added `n_jobs=None` parameter to `GAOptimizer.__init__`
- Added `_get_n_jobs()` method with bounds validation:
  - Raises `ValueError` if `n_jobs < 1`
  - Prints warning and clamps to `cpu_count()` if `n_jobs > cpu_count()`
  - Falls back to `multiprocessing.cpu_count()` when `n_jobs is None`
- Updated `multiprocessing.Pool()` to `multiprocessing.Pool(processes=self._get_n_jobs())`

**`EvoFlow/cli.py`:**
- Added `--n_jobs` CLI argument (`type=int`, `default=None`)
- Passes `n_jobs=args.n_jobs` to `run_evoflow()` which forwards to `GAOptimizer`

**`AGENTS.md`:**
- Updated Run Application example with `--n_jobs` flag
- Added Architecture Notes documenting `n_jobs` parameter and validation

**`README.md`:**
- Added `--n_jobs` row to Arguments table

### Technical Details
- `_get_n_jobs()` provides input validation to prevent DoS from extreme values
- The `**kwargs` pass-through in `run_evoflow()` and `GAOptimizer` already existed, so CLI integration was straightforward
- All 27 existing tests pass
- `ruff format` and `ruff check` both pass
- Security audit passed (bandit clean, no vulnerabilities)

### Verification Checklist
- [x] Security audit: SECURE
- [x] Linting: ruff check passed
- [x] Formatting: ruff format passed
- [x] Tests: 27/27 passed
- [x] User approved
