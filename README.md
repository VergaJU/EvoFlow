# EvoFlow

![Passed tests](https://github.com/VergaJU/EvoFlow/actions/workflows/ci.yml/badge.svg)

EvoFlow is a Python-based tool designed for biological network analysis and optimization. It combines graph theory (MaxFlow min-cut) with Genetic Algorithms (GA) to identify critical nodes in gene regulatory networks.

By integrating gene expression data (from `scanpy` AnnData objects) with signaling networks, EvoFlow identifies the optimal set of nodes to perturb in order to modulate signal flow from specific ligands to target genes.

## Features

- **Biological Network Construction**: builds directed graphs from ligand-receptor interactions and signaling pathways.
- **Data Integration**: Incorporates single-cell gene expression data (`.h5ad`) to weight network capacities.
- **Flow Analysis**: Computes Maximum Flow from ligands (sources) to target genes (sinks).
- **Genetic Algorithm Optimization**: Uses `deap` to find the optimal set of nodes to "knock out" to minimize flow (or other objectives) while preserving specific network structures.
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
| Expression | `-f`, `--file` | Path to `.h5ad` (Scanpy) file containing expression data |
| Regulons | `-r`, `--regulons` | Path to regulons CSV file |
| Receptors | `-b`, `--receptors` | Pipe-separated list of receptor genes (e.g. "R1\|R2") |
| Mediators | `-m`, `--mediators` | Comma-separated list of mediator genes |
| Ligands | `-s`, `--ligands` | CSV file containing ligand genes (must have column 'x') |
| Label | `-l`, `--label` | Column name in `adata.obs` for cell type grouping |
| Cell Type | `-c`, `--celltype` | Specific cell type to analyze from the label column |
| Generations | `--generations` | Number of GA generations (default: 10000) |
| Population | `--pop_size` | Population size for GA (default: 1000) |
| Early Stop | `--early_stop` | Generations without improvement before stopping (default: 100) |
| Delta Penalty | `--delta_penalty` | Apply delta penalty (default: False) |
| Repair | `--repair` | Repair invalid individuals (default: True) |
| Norm Hypervolume | `--norm_hypervolume` | Normalize hypervolume calculation (default: False) |
| Dynamic Rates | `--dynamic_rates` | Use dynamic mutation/crossover rates (default: True) |

### Python API

You can also run EvoFlow directly from your Python scripts:

```python
from EvoFlow.cli import run_evoflow

run_evoflow(
    network="path/to/network.txt",
    annotation="path/to/annotation.txt",
    file="path/to/data.h5ad",
    regulons="path/to/regulons.csv",
    receptors_raw="GeneA|GeneB",
    mediators_raw="GeneC,GeneD",
    ligands_file="path/to/ligands.csv",
    label="cell_type_column",
    celltype="TargetCellType",
    generations=1000,
    population_size=500,
    # Optional arguments
    early_stop=100,
    repair=True
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
├── biology.py       # Biological data loading, expression normalization, and graph construction
├── core.py          # Core Genetic Algorithm (DEAP) and MaxFlow (NetworkX) logic
├── visualization.py # Plotting utilities
└── cli.py           # Command-line entry point
```

### Architecture

1.  **Preprocessing**: The `BioNetworkPreparer` (`biology.py`) loads data, calculates "percolation" values (normalized gene expression), and builds the initial weighted graph.
2.  **Topology Setup**: "SuperSource" and "SuperSink" nodes are added to model the flow from Ligands -> Receptors -> Mediators -> Targets.
3.  **Active Subgraph**: The system extracts only the active subgraph (paths actually connecting sources to sinks) for optimization.
4.  **Optimization**: The `GAOptimizer` (`core.py`) runs a genetic algorithm to select nodes for removal, optimizing for objectives like reducing flow capacity while minimizing the number of removed nodes.

## Testing

The project includes a `tests/` suite using `pytest`.

To run tests:
```bash
pytest
```

## Contributing

See [AGENTS.md](AGENTS.md) for specific guidelines for AI agents and contributors working on this codebase.
