import argparse
import os
import time
import pandas as pd
import warnings
from EvoFlow.biology import BioNetworkPreparer
from EvoFlow.core import GAOptimizer, extract_active_subgraph
from EvoFlow.visualization import plot_optimization_history, plot_node_proportions, plot_node_attribute, plot_edge_attribute, save_network





def run_evoflow(network, annotation, file, regulons, receptors_raw, mediators_raw, 
                ligands_file, label, celltype, generations=10000, pop_size=1000):
    """
    This function contains all your logic and can be called from other scripts.
    """
    warnings.filterwarnings("ignore")
    
    # Process Inputs (Logic moved from main)
    receptors = receptors_raw.split('|')
    mediators = mediators_raw.split(',')
    mediators_list = mediators + receptors
    
    ligands_df = pd.read_csv(ligands_file)
    ligands = ligands_df.x.values.tolist()
    
    filename_prefix = str.split(str.split(os.path.basename(file), ".")[0], "tmp")[0]

    # --- Biology Phase ---
    preparer = BioNetworkPreparer()
    preparer.load_data(network, annotation, file, regulons)

    preparer.filter_expression_data(args.label, args.celltype)
    preparer.compute_percolation()
    preparer.build_base_network()
    
    # Update Capacities
    preparer.update_edge_capacities(method='multiplicative')
    
    # Identify Targets
    target_genes = preparer.identify_targets(mediators_list)
    
    # Visualization (Pre-optimization)
    plot_node_proportions(preparer.annotation, filename_prefix)
    plot_node_attribute(preparer.G, filename_prefix, attribute='weight', show=False)
    plot_edge_attribute(preparer.G, filename_prefix, 'capacity', show=False)
    # Prepare Topology (Add SuperSource/Sink)
    sources, sinks = preparer.prepare_flow_topology(ligands, target_genes)
    
    G = preparer.get_graph()
    
    # Extract Subgraph (Optimization on active subgraph only)
    # Original script does this logic:
    # 1. Calculate Flow on G
    # 2. Extract Subgraph
    # 3. Calculate Flow on Subgraph
    # 4. Use Subgraph for GA
    
    print(f"{time.ctime()} - Extracting subgraph between source and sink")
    SG = extract_active_subgraph(G, 'SuperSource', 'SuperSink')
    
    print(f"{time.ctime()} - Original graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"{time.ctime()} - Subgraph: {SG.number_of_nodes()} nodes, {SG.number_of_edges()} edges")
    
    if SG.number_of_nodes() == 0:
        print("Error: Empty subgraph. No path between sources and sinks.")
        return

    # --- Optimization Phase ---
    print(f"{time.ctime()} - Initializing Genetic Algorithm")
    
    optimizer = GAOptimizer(SG, sources, sinks, receptors, target_genes)
    
    # Run GA
    best_nodes_removed, logs = optimizer.run(
        population_size=args.pop_size, 
        generations=args.generations,
        early_stop_gen=100
    )
    
    # --- Results ---
    # Save Best Candidates
    pd.DataFrame(best_nodes_removed).to_csv(filename_prefix + "best_candidates.csv", header=False, index=False)
    
    # Plot History
    plot_optimization_history(logs, filename_prefix)
    
    # Save Final Network
    # Apply removals to SG to get final result
    Final_G = SG.copy()
    save_network(Final_G, filename_prefix)
    
    print(f"{time.ctime()} - Optimization Complete.")
    return filename_prefix # Or return the Final_G if you want to use it in Python



def main():
    # This only runs when calling from CLI
    parser = argparse.ArgumentParser(description="EvoFlow: Network Analysis and GA Optimization")
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
        pop_size=args.pop_size
    )

if __name__ == "__main__":
    main()
