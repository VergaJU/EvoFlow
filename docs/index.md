# EvoFlow

**EvoFlow** finds optimal node knock-outs in gene regulatory networks by
combining maximum-flow / min-cut analysis with a multi-objective genetic
algorithm (NSGA-II). Starting from a ligand-receptor signalling network, it
scores candidate removals by how much they reduce the flow of signal from
ligands (sources) to target genes (sinks), how many nodes they remove, and how
much redundant strongly-connected structure they break.

```{image} https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue
:alt: Python versions
```

## Features

- **Biological network construction** from ligand-receptor and signalling
  networks, with `SuperSource`/`SuperSink` flow topology.
- **Expression integration** from single-cell (`.h5ad`) or bulk
  (`.csv`/`.tsv`) data, with configurable per-gene statistics and
  normalization.
- **Maximum-flow analysis** on an integer CSR representation for speed.
- **Genetic algorithm optimization** with caching, optional multiprocessing,
  repair, adaptive rates and selectable source/sink locking.
- **GAM parameter estimation** that predicts a good population size and early
  stopping patience from network features.
- **Visualization** of optimization history, node/edge attribute
  distributions and the final network (GraphML, Cytoscape JSON).

## Installation

EvoFlow is distributed on PyPI as **`evoflow-bio`** (the import name is
`EvoFlow`):

```bash
pip install evoflow-bio
```

Optional extras:

```bash
pip install "evoflow-bio[docs]"      # Sphinx toolchain to build these docs
pip install "evoflow-bio[vignette]"  # nichenetpy + pertpy for the vignette
```

Verify the install:

```bash
evoflow --help
```

### Container

A reproducible Jupyter image with EvoFlow, NicheNetPy and pertpy
pre-installed is published on the GitHub Container Registry:

```bash
docker run --rm -p 8888:8888 ghcr.io/vergaju/evoflow:latest
```

See {doc}`container` for mounting a working directory and opening the
vignette.

## Quick start

```python
from EvoFlow.cli import run_evoflow

run_evoflow(
    network="network.tsv",
    annotation="annotation.tsv",
    expression_file="expression.h5ad",
    receptors_raw="CD3E|CD28",
    mediators_raw="TBX21,IFNG",
    ligands_file="ligands.csv",
    targets="GZMB,PRF1",
    label="cell_type",  # h5ad only; optional with celltype
    celltype="NK",
    generations=1000,
)
```

The command-line interface exposes the same pipeline:

```bash
evoflow \
    --network network.tsv \
    --annotation annotation.tsv \
    --file expression.h5ad \
    --regulons regulons.csv \
    --receptors "CD3E|CD28" \
    --mediators "TBX21,IFNG" \
    --ligands ligands.csv \
    --label cell_type \
    --celltype NK
```

Results (JSON/YAML knock-out lists, normalization table, population/log
pickles, GraphML/Cytoscape networks and PNG plots) are written to the current
working directory.

## Contents

```{toctree}
:maxdepth: 2
:hidden:

install
api
container
vignettes/index
```

```{toctree}
:maxdepth: 1
:caption: Project links
:hidden:

GitHub <https://github.com/VergaJU/EvoFlow>
Changelog <https://github.com/VergaJU/EvoFlow/blob/master/CHANGELOG.md>
```
