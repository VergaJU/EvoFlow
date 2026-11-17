import argparse
import os
import time
import warnings

import pandas as pd

from EvoFlow import utils
from EvoFlow.biology import EXPRESSION_NORMS, EXPRESSION_STATS, BioNetworkPreparer
from EvoFlow.core import GAOptimizer, extract_active_subgraph
from EvoFlow.visualization import (
    plot_edge_attribute,
    plot_node_attribute,
    plot_node_proportions,
    plot_optimization_history,
    save_network,
)


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
    lock_targets: int = 0,
    expression_stat: str = "nonzero_ratio",
    expression_norm: str = "none",
    expression_transpose: bool = False,
    generations=10000,
    population_size=None,
    early_stop_gen=None,
    guess_parameters=True,
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
        lock_targets (int): Bitmask controlling which original nodes are locked as sources/sinks.
            0 (default): No locking.
            1: Original source nodes are locked into final_sources.
            2: Original sink nodes are locked into final_sinks.
            3: Both source and sink nodes are locked.
        expression_stat (str): Per-gene statistic for percolation, one of
            EXPRESSION_STATS (default: "nonzero_ratio").
        expression_norm (str): Normalization applied before the statistic
            (ignored for "nonzero_ratio"), one of EXPRESSION_NORMS (default: "none").
        expression_transpose (bool): Orient genes-as-rows csv/tsv bulk expression
            files to samples x genes (default: False; not applicable to h5ad).
        generations (int): Number of GA generations (default 10000).
        population_size (int, optional): GA population size. Must be given together
            with early_stop_gen; providing either disables GAM guessing.
        early_stop_gen (int, optional): Generations without improvement before
            stopping. Must be given together with population_size.
        guess_parameters (bool): Estimate GA parameters with the GAM when
            population_size/early_stop_gen are omitted (default True).
        return_results (bool): If True, return results; if False, return filename_prefix.
        **kwargs: Additional keyword arguments passed to GAOptimizer.

    Returns:
        If ``return_results`` is True: ``(best_nodes_removed, logs, pop, norm)``.
        Otherwise: ``filename_prefix`` (str), or ``None`` when the network
        preparation or subgraph extraction fails.

    Raises:
        ValueError: If ``population_size``/``early_stop_gen`` are only
            half-specified, if ``expression_stat``/``expression_norm`` are
            invalid, or if only one of ``label``/``celltype`` is given.

    Examples:
        >>> from EvoFlow.cli import run_evoflow
        >>> run_evoflow(  # doctest: +SKIP
        ...     network="network.tsv",
        ...     annotation="annotation.tsv",
        ...     expression_file="data.h5ad",
        ...     receptors_raw="CD3E|CD28",
        ...     mediators_raw="TBX21,IFNG",
        ...     ligands_file="ligands.csv",
        ...     targets="GZMB,PRF1",
        ...     population_size=50,
        ...     early_stop_gen=10,
        ...     generations=20,
        ... )
    """
    if (population_size is None) != (early_stop_gen is None):
        raise ValueError(
            "population_size and early_stop_gen must be given together, "
            "or both omitted to let the GAM guess."
        )
    if population_size is not None:
        guess_parameters = False
    if expression_stat not in EXPRESSION_STATS:
        raise ValueError(
            f"expression_stat must be one of {EXPRESSION_STATS}, got {expression_stat!r}"
        )
    if expression_norm not in EXPRESSION_NORMS:
        raise ValueError(
            f"expression_norm must be one of {EXPRESSION_NORMS}, got {expression_norm!r}"
        )
    if (label is None) != (celltype is None):
        raise ValueError(
            "label and celltype must be provided together (both for h5ad cell "
            "filtering, or both omitted)"
        )

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
            lock_targets=lock_targets,
            expression_stat=expression_stat,
            expression_norm=expression_norm,
            expression_transpose=expression_transpose,
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
        population_size=population_size,
        early_stop_gen=early_stop_gen,
        guess_parameters=guess_parameters,
        generations=generations,
        **kwargs,
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
    """Parse command-line arguments and run the EvoFlow pipeline.

    Exposed as the ``evoflow`` console script. See the README (or the online
    documentation) for the meaning of each flag.
    """
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
    parser.add_argument(
        "-l",
        "--label",
        default=None,
        help="obs column for cell type filtering (h5ad only; requires --celltype)",
    )
    parser.add_argument(
        "-c",
        "--celltype",
        default=None,
        help="Cell type to filter by (h5ad only; requires --label)",
    )
    parser.add_argument(
        "--expression_stat",
        choices=EXPRESSION_STATS,
        default="nonzero_ratio",
        help="Per-gene statistic used for percolation (default: nonzero_ratio)",
    )
    parser.add_argument(
        "--expression_norm",
        choices=EXPRESSION_NORMS,
        default="none",
        help="Expression normalization applied before --expression_stat; ignored "
        "for nonzero_ratio (default: none)",
    )
    parser.add_argument(
        "--expression_transpose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Expression file has genes as rows (bulk csv/tsv with gene IDs as "
        "first column); transpose to samples x genes (default: False)",
    )
    parser.add_argument("--generations", type=int, default=10000)
    parser.add_argument(
        "--pop_size",
        type=int,
        default=None,
        help="GA population size (omit to let the GAM guess; "
        "must be given together with --early_stop)",
    )
    parser.add_argument(
        "--early_stop",
        type=int,
        default=None,
        help="Generations without improvement before stopping "
        "(omit to let the GAM guess; must be given together with --pop_size)",
    )
    parser.add_argument(
        "--guess_parameters",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Estimate GA parameters with the GAM (default: True). Providing "
        "--pop_size/--early_stop disables guessing automatically.",
    )
    parser.add_argument(
        "--delta_penalty",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply delta penalty (default: False)",
    )
    parser.add_argument(
        "--repair",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Repair invalid individuals (default: True)",
    )
    parser.add_argument(
        "--norm_hypervolume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Normalize hypervolume calculation (default: False)",
    )
    parser.add_argument(
        "--dynamic_rates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use dynamic mutation/crossover rates (default: True)",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=None,
        help="Number of parallel jobs for multiprocessing (default: all CPUs)",
    )
    parser.add_argument(
        "--lock_targets",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="Lock original source/sink nodes: 0=none, 1=sources, 2=sinks, 3=both (default: 0)",
    )

    args = parser.parse_args()

    if (args.pop_size is None) != (args.early_stop is None):
        parser.error(
            "--pop_size and --early_stop must be provided together "
            "(or both omitted to let the GAM guess)"
        )
    if not args.guess_parameters and args.pop_size is None:
        parser.error("--no-guess_parameters requires both --pop_size and --early_stop")
    if (args.label is None) != (args.celltype is None):
        parser.error(
            "--label and --celltype must be provided together "
            "(both for h5ad cell filtering, or both omitted)"
        )

    # Call the new function using arguments from the CLI
    run_evoflow(
        network=args.network,
        annotation=args.annotation,
        expression_file=args.file,
        regulons=args.regulons,
        receptors_raw=args.receptors,
        mediators_raw=args.mediators,
        ligands_file=args.ligands,
        label=args.label,
        celltype=args.celltype,
        lock_targets=args.lock_targets,
        expression_stat=args.expression_stat,
        expression_norm=args.expression_norm,
        expression_transpose=args.expression_transpose,
        generations=args.generations,
        population_size=args.pop_size,
        early_stop_gen=args.early_stop,
        guess_parameters=args.guess_parameters,
        delta_penalty=args.delta_penalty,
        repair=args.repair,
        norm_hypervolume=args.norm_hypervolume,
        dynamic=args.dynamic_rates,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
