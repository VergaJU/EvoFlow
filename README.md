# EvoFlow

![Passed tests](https://github.com/VergaJU/EvoFlow/actions/workflows/ci.yml/badge.svg)

EvoFlow is a Python-based tool designed for biological network analysis and optimization. It combines graph theory (MaxFlow min-cut) with Genetic Algorithms (GA) to identify critical nodes in gene regulatory networks.

By integrating gene expression data (from `scanpy` AnnData objects) with signaling networks, EvoFlow identifies the optimal set of nodes to perturb in order to modulate signal flow from specific ligands to target genes.

## Features

- **Biological Network Construction**: builds directed graphs from ligand-receptor interactions and signaling pathways.
- **Data Integration**: Incorporates expression data — single-cell (`.h5ad`) or bulk (`.csv`/`.tsv`) — to weight network capacities, with configurable normalization (CPM, log1p, min-max) and per-gene statistics.
- **Flow Analysis**: Computes Maximum Flow from ligands (sources) to target genes (sinks).
- **Genetic Algorithm Optimization**: Uses `deap` to find the optimal set of nodes to "knock out" to minimize flow (or other objectives) while preserving specific network structures.
- **GAM Parameter Estimation**: Uses a Generalized Additive Model (GAM) to automatically estimate optimal GA population size and early stopping patience from network features, removing the need for manual tuning.
- **Visualization**: Generates plots for optimization history, weight distributions, and network topology.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/jacopo/EvoFlow.git
   cd EvoFlow
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

EvoFlow is run as a Python module via the Command Line Interface (CLI).

### Basic Command

```bash
python -m EvoFlow.cli \
    --network path/to/network.txt \
    --annotation path/to/annotation.txt \
    --file path/to/data.h5ad \
    --regulons path/to/regulons.csv \
    --receptors "GeneA|GeneB" \
    --mediators "GeneC,GeneD" \
    --ligands path/to/ligands.csv \
    --label "cell_type_column" \
    --celltype "TargetCellType"
```

### Arguments

| Argument | Flag | Description |
|----------|------|-------------|
| Network | `-n`, `--network` | Path to the network file (TSV: source, target, weight, type) |
| Annotation | `-a`, `--annotation` | Path to annotation file (TSV mapping genes to types) |
| Expression | `-f`, `--file` | Path to expression matrix: `.h5ad` (Scanpy) or `.csv`/`.tsv` (bulk) |
| Regulons | `-r`, `--regulons` | Path to regulons CSV file |
| Receptors | `-b`, `--receptors` | Pipe-separated list of receptor genes (e.g. "R1\|R2") |
| Mediators | `-m`, `--mediators` | Comma-separated list of mediator genes |
| Ligands | `-s`, `--ligands` | CSV file containing ligand genes (must have column 'x') |
| Label | `-l`, `--label` | Column name in `adata.obs` for cell type filtering (h5ad only; optional, requires `--celltype`) |
| Cell Type | `-c`, `--celltype` | Specific cell type to analyze from the label column (h5ad only; optional, requires `--label`) |
| Expression Stat | `--expression_stat` | Per-gene statistic used for percolation: `nonzero_ratio` (default), `mean`, `median`, `p90`, `max` |
| Expression Norm | `--expression_norm` | Normalization applied before `--expression_stat` (ignored for `nonzero_ratio`): `none` (default), `cpm`, `log1p`, `minmax` |
| Expression Transpose | `--expression_transpose` / `--no-expression_transpose` | Bulk `.csv`/`.tsv` files have genes as rows (gene IDs in the first column); transpose to samples × genes (default: False; not applicable to `.h5ad`) |
| Generations | `--generations` | Number of GA generations (default: 10000) |
| Population | `--pop_size` | GA population size; omit to let the GAM guess — must be given together with `--early_stop` |
| Early Stop | `--early_stop` | Generations without improvement before stopping; omit to let the GAM guess — must be given together with `--pop_size` |
| Delta Penalty | `--delta_penalty` / `--no-delta_penalty` | Apply delta penalty (default: False) |
| Repair | `--repair` / `--no-repair` | Repair invalid individuals (default: True) |
| Norm Hypervolume | `--norm_hypervolume` / `--no-norm_hypervolume` | Normalize hypervolume calculation (default: False) |
| Dynamic Rates | `--dynamic_rates` / `--no-dynamic_rates` | Use dynamic mutation/crossover rates (default: True) |
| Parallel Jobs | `--n_jobs` | Number of parallel jobs for multiprocessing (default: all CPUs) |
| Lock Targets | `--lock_targets {0,1,2,3}` | Lock original source (1) / sink (2) / both (3) nodes against GA removal (default: 0) |
| Guess Params | `--guess_parameters` / `--no-guess_parameters` | Estimate GA parameters using GAM (default: True); providing `--pop_size`/`--early_stop` disables guessing automatically |

### Python API

You can also run EvoFlow directly from your Python scripts:

```python
from EvoFlow.cli import run_evoflow

run_evoflow(
    network="path/to/network.txt",
    annotation="path/to/annotation.txt",
    expression_file="path/to/data.h5ad",
    regulons="path/to/regulons.csv",
    receptors_raw="GeneA|GeneB",
    mediators_raw="GeneC,GeneD",
    ligands_file="path/to/ligands.csv",
    label="cell_type_column",
    celltype="TargetCellType",
    generations=1000,
    # population_size + early_stop_gen are given together and
    # automatically disable GAM guessing; omit both to let the GAM guess
    population_size=500,
    early_stop_gen=100,
    # Optional arguments
    repair=True,
    # Expression handling (bulk files with genes as rows):
    # expression_transpose=True,
    # expression_stat="mean",       # nonzero_ratio | mean | median | p90 | max
    # expression_norm="cpm",        # none | cpm | log1p | minmax (ignored for nonzero_ratio)
)
```

## Outputs

EvoFlow generates several output files, prefixed with the input filename:

**Data:**
- `*_ga_results.json` / `.yaml`: Best candidate nodes for removal found by the GA.
- `*_norm.tsv`: Normalization data.
- `*_population.pkl`: Final GA population.
- `*_logs.pkl`: Optimization logs.

**Networks:**
- `*_network.graphml`: Final optimized network in GraphML format.
- `*_network.cyjs`: Final optimized network in Cytoscape JSON format.

**Plots:**
- `*_optimization_values.png`: Evolution of GA metrics (Flow, Node count, etc.) over generations.
- `*_nodes_proportions.png`: Bar chart of node types.
- `*_percolation.png`: Rank plot and histogram of node percolation values.
- `*_edge_capacity.png`: Rank plot and histogram of edge capacities.
- `*_edge_weight.png`: Rank plot and histogram of edge weights.

## Project Structure

The project has been refactored into a modular package structure:

```text
EvoFlow/
├── biology.py               # Biological data loading, expression normalization, and graph construction
├── core.py                  # Core Genetic Algorithm (DEAP) and MaxFlow (NetworkX) logic
├── features_extraction.py   # Graph feature extraction (density, degree distribution, etc.)
├── gam_preprocessing.py     # Data preprocessing pipeline for GAM training
├── model.py                 # GAM model wrapper (training, prediction, parameter ranking)
├── train.py                 # CLI to train the GAM model from collected data
├── visualization.py         # Plotting utilities
├── models/                  # Pre-trained models and configuration
│   ├── config.yaml          # GAM feature configuration
│   ├── model.pkl            # Trained LinearGAM
│   └── training_data.csv    # Training dataset
└── cli.py                   # Command-line entry point
```

### Architecture

1.  **Preprocessing**: The `BioNetworkPreparer` (`biology.py`) loads data, reduces expression to a per-gene "percolation" value (configurable statistic and normalization), and builds the initial weighted graph.
2.  **Topology Setup**: "SuperSource" and "SuperSink" nodes are added to model the flow from Ligands -> Receptors -> Mediators -> Targets.
3.  **Active Subgraph**: The system extracts only the active subgraph (paths actually connecting sources to sinks) for optimization.
4.  **Parameter Estimation (optional)**: The `GAOptimizer` extracts graph features (via `NetworkFeatures` in `features_extraction.py`) and uses a pre-trained GAM model (in `model.py`) to recommend optimal `population_size` and `early_stop_gen` values.
5.  **Optimization**: The `GAOptimizer` (`core.py`) runs a genetic algorithm to select nodes for removal, optimizing for objectives like reducing flow capacity while minimizing the number of removed nodes.

## Testing

The project includes a `tests/` suite using `pytest`.

To run tests:
```bash
pytest
```

## Contributing

See [AGENTS.md](AGENTS.md) for specific guidelines for AI agents and contributors working on this codebase.
