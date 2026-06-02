# EvoFlow Logbook

## Session: Implement Orchestrator-Level Caching

**Date:** 2026-06-02
**Branch:** `feature_orchestrator_cache_core_002`
**Commit:** `2a4d16e`

### Summary
Replaced the `multiprocessing.Manager.dict()` worker-level cache with an orchestrator-level `SmartCacheMap` that caches evaluation results in the main process, eliminating IPC overhead on cache hits.

### Changes Made

**`EvoFlow/core.py`:**
- Added `SmartCacheMap` class — a callable wrapper that:
  - Checks a local `dict` (`tuple(ind)` keys) before dispatching to `pool.map`
  - Sends only cache misses to workers
  - Updates cache with computed results
  - Works with both multiprocessing (`pool.map`) and sequential (`for` loop) modes
- Simplified `evaluate()` — removed `shared_cache` parameter and all cache read/write logic
- Simplified `constrained_evaluate()` — removed `shared_cache` parameter
- Updated `__init__()` baseline `evaluate()` call — removed `shared_cache={}`
- Refactored `run()` — uses `functools.partial` to bind evaluation parameters, `SmartCacheMap` replaces the old `Manager.dict()` + `pool.map` pattern
- Updated `finally` block — removed `manager.shutdown()`

**`tests/test_core.py`** and **`tests/test_core_advanced.py`:**
- Removed stale `shared_cache=cache` arguments from test calls

**`AGENTS.md`:**
- Added Architecture Notes documenting `SmartCacheMap` caching and `functools.partial` usage

### Technical Details
- Cache key: `tuple(ind)` — the binary representation of the individual
- Cache scope: Created fresh inside each `run()` call, lives for its duration
- Cache hit: Returns stored result instantly — no IPC, no pickling
- Cache miss: Dispatches to multiprocessing pool (or sequential loop), stores result on return
- `functools.partial` binds graph, nodes, and sink/source parameters into a picklable callable

### Verification Checklist
- [x] Security audit: SECURE (bandit clean, no DoS via cache growth — scoped per run)
- [x] Linting: ruff check passed
- [x] Formatting: ruff format passed
- [x] Tests: 27/27 passed
- [x] User approved
