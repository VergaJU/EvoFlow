# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-11-16

First public release, distributed on PyPI as `evoflow-bio` (import name
`EvoFlow`).

### Added

- Biological network construction from ligand-receptor and signalling networks,
  with SuperSource/SuperSink flow topology and configurable locking of original
  source/sink nodes (`lock_targets` bitmask).
- Maximum-flow analysis on an integer CSR representation
  (`scipy.sparse.csgraph.maximum_flow`) with active-subgraph extraction.
- Multi-objective genetic algorithm (`deap`, NSGA-II) that selects nodes to
  remove while trading off flow reduction, removal cost and redundancy, with
  optional repair, delta penalty, dynamic mutation/crossover rates and
  multiprocessing (`n_jobs`).
- Expression integration from `.h5ad` (single-cell) and `.csv`/`.tsv` (bulk)
  files, with per-gene statistics (`nonzero_ratio`, `mean`, `median`, `p90`,
  `max`) and normalization (`none`, `cpm`, `log1p`, `minmax`), including
  genes-as-rows bulk files via `expression_transpose`.
- GAM-based estimation of GA population size and early-stopping patience from
  network features, with a bundled pre-trained model and retraining CLI
  (`python -m EvoFlow.train`).
- Command-line interface (`evoflow`) and Python API (`EvoFlow.cli.run_evoflow`).
- Visualization helpers for optimization history, node/edge attribute
  distributions and network export (GraphML, Cytoscape JSON).
  `plot_graph` accepts `layout_kwargs`, `node_size`, `font_size`,
  `with_labels` and `edge_alpha`, so dense networks can be drawn legibly and
  reproducible layouts (for example `spring` with a fixed seed) are possible.
- Full Sphinx API documentation on Read the Docs, including a pre-executed
  end-to-end vignette.
- Jupyter Docker image published to GHCR (`ghcr.io/vergaju/evoflow`) based on
  the Jupyter SciPy stack, with NicheNetPy and pertpy pre-installed for
  upstream network inference and example data.
- GitHub Actions workflows for CI, GHCR image publishing and PyPI Trusted
  Publishing. The container workflow also builds (without pushing) on pull
  requests, and publishes `v<version>` tags in addition to semver tags.

### Changed

- Package metadata now uses PEP 639 license expression, an explicit package
  discovery (`EvoFlow*` only) so tests and docs are no longer shipped in the
  wheel, and ships the GAM training data with the package.
- Documentation and examples now install from PyPI with `pip install
  evoflow-bio`.
- The vignette is organised as two complete analyses on the same NK-cell data:
  the IFN-β response (IFNAR1/2 → interferon-stimulated genes) and the NK
  activation/effector program (activating and inhibitory receptors → granule and
  cytokine effectors). Each analysis writes its own network prefix, so the two
  runs no longer overwrite each other.
- CI installs the package (`pip install -e ".[dev]"`) instead of only its
  requirements, so entry points and packaged model data are tested. A
  `.dockerignore` keeps the image build context small, and generated Sphinx
  stubs and vignette scratch files are ignored by git.

[1.0.0]: https://github.com/VergaJU/EvoFlow/releases/tag/v1.0.0
