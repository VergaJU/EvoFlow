import argparse
import os
import time
import pandas as pd
import warnings
from EvoFlow.biology import BioNetworkPreparer
from EvoFlow.core import GAOptimizer, extract_active_subgraph
from EvoFlow.visualization import (
    plot_optimization_history,
    plot_node_proportions,
    plot_node_attribute,
    plot_edge_attribute,
    save_network,
)
from EvoFlow import utils


def run_evoflow(
    network,
    annotation,
    expression_file,
    receptors_raw: str = None,
    mediators_raw: str = None,
    ligands_file: str = None,
    label: str = None,
    celltype: str = None,
    regulons: str = None,
    targets: str = None,
    generations=10000,
    population_size=1000,
    return_results=False,
    **kwargs,
):
    """
    Orchestrates the full EvoFlow pipeline: biological network preparation followed by
    genetic algorithm optimization. Can be called from other scripts or CLI.

    Args:
        network (str): Path to network file.
        annotation (str): Path to annotation file.
        expression_file (str): Path to expression data file (.h5ad, .csv, or .tsv).
        receptors_raw (str, optional): Pipe-delimited list of receptors.
        mediators_raw (str, optional): Comma-delimited list of mediators.
        ligands_file (str): Path to ligands CSV file (must contain 'x' column).
        label (str, optional): Column name in AnnData obs for cell type filtering.
        celltype (str, optional): Cell type to filter by.
        regulons (str, optional): Path to regulons file for target identification.
        targets (str, optional): Comma-delimited list of target genes (if regulons not used).
        generations (int): Number of GA generations (default 10000).
        population_size (int): GA population size (default 1000).
        return_results (bool): If True, return results; if False, return filename_prefix.
        **kwargs: Additional keyword arguments passed to GAOptimizer.

    Returns:
        If return_results is True: (best_nodes_removed, logs, pop, norm)
        Otherwise: filename_prefix (str)
    """
    warnings.filterwarnings("ignore")

    # Process Inputs (Logic moved from main)
    if receptors_raw and mediators_raw:
        receptors = receptors_raw.split("|")
        mediators = mediators_raw.split(",")
        mediators_list = mediators + receptors
    elif receptors_raw:
        receptors = receptors_raw.split("|")
        mediators_list = receptors
    elif mediators_raw:
        mediators = mediators_raw.split(",")
        mediators_list = mediators
    else:
        mediators_list = []

    ligands_df = pd.read_csv(ligands_file)
    ligands = ligands_df.x.values.tolist()

    filename_prefix = str.split(
        str.split(os.path.basename(expression_file), ".")[0], "tmp"
    )[0]

    # Parse targets if provided as string
    targets_list = targets.split(",") if targets else None

    # --- Biology Phase ---
    preparer = BioNetworkPreparer()
    try:
        G, sources, sinks = preparer.prepare_network(
            network_path=network,
            annotation_path=annotation,
            expression_path=expression_file,
            ligands=ligands,
            mediators_list=mediators_list,
            label=label,
            celltype=celltype,
            regulons_path=regulons,
            targets=targets_list,
        )
    except ValueError as e:
        print(f"Error during network preparation: {e}")
        return

    # Visualization (Pre-optimization)
    plot_node_proportions(preparer.annotation, filename_prefix)
    plot_node_attribute(G, filename_prefix, attribute="percolation", show=False)
    plot_edge_attribute(G, filename_prefix, "capacity", show=False)
    plot_edge_attribute(G, filename_prefix, "weight", show=False)

    # Extract Subgraph (Optimization on active subgraph only)
    # Original script does this logic:
    # 1. Calculate Flow on G
    # 2. Extract Subgraph
    # 3. Calculate Flow on Subgraph
    # 4. Use Subgraph for GA

    print(f"{time.ctime()} - Extracting subgraph between source and sink")
    SG = extract_active_subgraph(G, "SuperSource", "SuperSink")

    print(
        f"{time.ctime()} - Original graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
    )
    print(
        f"{time.ctime()} - Subgraph: {SG.number_of_nodes()} nodes, {SG.number_of_edges()} edges"
    )

    if SG.number_of_nodes() == 0:
        print("Error: Empty subgraph. No path between sources and sinks.")
        return

    # --- Optimization Phase ---
    print(f"{time.ctime()} - Initializing Genetic Algorithm")

    optimizer = GAOptimizer(G=SG, sources=sources, sinks=sinks, **kwargs)

    # Run GA
    best_nodes_removed, logs, pop, norm = optimizer.run(
        population_size=population_size, generations=generations, **kwargs
    )

    # --- Results ---
    # Save Best Candidates
    utils.save_to_json(
        data=best_nodes_removed, filename=filename_prefix + "_ga_results.json"
    )
    utils.save_to_yaml(
        data=best_nodes_removed, filename=filename_prefix + "_ga_results.yaml"
    )
    utils.save_norm(norm=norm, filename=filename_prefix + "_norm.tsv")
    utils.save_population(population=pop, filename=filename_prefix + "_population.pkl")
    utils.save_logs(logs=logs, filename=filename_prefix + "_logs.pkl")
    # Plot History
    plot_optimization_history(logs, filename_prefix)

    # Save Final Network
    # Apply removals to SG to get final result
    Final_G = SG.copy()
    save_network(Final_G, filename_prefix)

    print(f"{time.ctime()} - Optimization Complete.")
    if return_results:
        return best_nodes_removed, logs, pop, norm
    return filename_prefix  # Or return the Final_G if you want to use it in Python


def main():
    # This only runs when calling from CLI
    parser = argparse.ArgumentParser(
        description="EvoFlow: Network Analysis and GA Optimization"
    )
    parser.add_argument("-n", "--network", required=True)
    parser.add_argument("-a", "--annotation", required=True)
    parser.add_argument("-f", "--file", required=True)
    parser.add_argument("-r", "--regulons", required=True)
    parser.add_argument("-b", "--receptors", required=True)
    parser.add_argument("-m", "--mediators", required=True)
    parser.add_argument("-s", "--ligands", required=True)
    parser.add_argument("-l", "--label", required=True)
    parser.add_argument("-c", "--celltype", required=True)
    parser.add_argument("--generations", type=int, default=10000)
    parser.add_argument("--pop_size", type=int, default=1000)
    parser.add_argument("--early_stop", type=int, default=100)
    parser.add_argument("--delta_penalty", type=bool, default=False)
    parser.add_argument("--repair", type=bool, default=True)
    parser.add_argument("--norm_hypervolume", type=bool, default=False)
    parser.add_argument("--dynamic_rates", type=bool, default=True)

    args = parser.parse_args()

    # Call the new function using arguments from the CLI
    run_evoflow(
        network=args.network,
        annotation=args.annotation,
        file=args.file,
        regulons=args.regulons,
        receptors_raw=args.receptors,
        mediators_raw=args.mediators,
        ligands_file=args.ligands,
        label=args.label,
        celltype=args.celltype,
        generations=args.generations,
        pop_size=args.pop_size,
    )


if __name__ == "__main__":
    main()
