import argparse
import os
import time
import pandas as pd
import warnings
from EvoFlow.biology import BioNetworkPreparer
from EvoFlow.core import GAOptimizer, extract_active_subgraph
from EvoFlow.visualization import plot_optimization_history, plot_node_proportions, plot_weights, save_network

def main():
    # Ignore warnings
    warnings.filterwarnings("ignore")
    
    # Argument Parser
    parser = argparse.ArgumentParser(description="EvoFlow: Network Analysis and GA Optimization")
    
    parser.add_argument("-n", "--network", help="Network from nichenet", required=True)
    parser.add_argument("-a", "--annotation", help="Annotation file", required=True)
    parser.add_argument("-f", "--file", help="h5ad file path", required=True)
    parser.add_argument("-r", "--regulons", help="Regulons file", required=True)
    parser.add_argument("-b", "--receptors", help="Receptors (pipe separated)", required=True) 
    parser.add_argument("-m", "--mediators", help="Mediators (comma separated)", required=True)
    parser.add_argument("-s", "--ligands", help="Ligands CSV file", required=True)
    parser.add_argument("-l", "--label", help="adata.obs label column", required=True)
    parser.add_argument("-c", "--celltype", help="Cell type of interest", required=True)
    parser.add_argument("--generations", help="Number of GA generations", type=int, default=10000)
    parser.add_argument("--pop_size", help="Population size", type=int, default=1000)

    args = parser.parse_args()
    
    # Process Inputs
    receptors = args.receptors.split('|')
    mediators = args.mediators.split(',')
    mediators_list = mediators + receptors
    
    # Load Ligands
    ligands_df = pd.read_csv(args.ligands)
    # Assuming 'x' column exists as per original script
    ligands = ligands_df.x.values.tolist()
    
    filename_prefix = str.split(str.split(os.path.basename(args.file), ".")[0], "tmp")[0]

    # --- Biology Phase ---
    preparer = BioNetworkPreparer()
    preparer.load_data(args.network, args.annotation, args.file, args.regulons)
    
    preparer.filter_expression_data(args.label, args.celltype)
    preparer.compute_percolation()
    preparer.build_base_network()
    
    # Update Capacities
    preparer.update_edge_capacities(method='multiplicative')
    
    # Identify Targets
    target_genes = preparer.identify_targets(mediators_list)
    
    # Visualization (Pre-optimization)
    plot_node_proportions(preparer.annotation, filename_prefix)
    plot_weights(preparer.net_df, filename_prefix)
    
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
    Final_G.remove_nodes_from(best_nodes_removed)
    save_network(Final_G, filename_prefix)
    
    print(f"{time.ctime()} - Optimization Complete.")

if __name__ == "__main__":
    main()
