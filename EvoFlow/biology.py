import pandas as pd
import numpy as np
import networkx as nx
import scanpy as sc
import scipy as sp
import os
import time

class BioNetworkPreparer:
    """
    Handles biological data loading, network construction, and capacity calculation
    based on gene expression data.
    """
    
    def __init__(self):
        self.G = None
        self.annotation = None
        self.adata = None
        self.regulons = None
        
    def load_data(self, network_path, annotation_path, h5ad_path, regulons_path=None):
        """
        Loads the raw input files.
        """
        print(f"{time.ctime()} - Reading files")
        # Load Network
        self.net_df = pd.read_csv(network_path, sep="\t")
        if "from" in self.net_df.columns:
            self.net_df = self.net_df.rename(columns={"from": "start"})
            
        # Load Annotation
        self.annotation = pd.read_csv(annotation_path, sep="\t", index_col="gene").squeeze()
        
        # Load Expression Data
        self.adata = sc.read(h5ad_path)
        
        # Load Regulons (Optional)
        if regulons_path:
            self.regulons = pd.read_csv(regulons_path)
            if 0 in self.regulons.index:
                self.regulons = self.regulons.drop(0)

    def filter_expression_data(self, label_col, cell_type):
        """
        Filters the AnnData object for a specific cell type.
        """
        self.adata = self.adata[self.adata.obs[label_col] == cell_type]

    def compute_percolation(self):
        """
        Computes the proportion of gene-expressing cells to be used as percolation values.
        """
        # Export count matrix and transform it to a Series with proportion of gene-expressing cells
        df = pd.DataFrame(sp.sparse.csr_matrix.todense(self.adata.X))
        df.columns = self.adata.var_names
        df = df.astype("bool").sum()
        
        # Normalize
        if (df.max() - df.min()) != 0:
            df = (df - df.min()) / (df.max() - df.min())
        else:
            df[:] = 0 # Handle edge case where all values are same

        # Create dictionary with proportion of gene-expressing cells
        # if the gene is not present in our dataframe, assign 0 to the gene
        self.percolation_values = {i: df.loc[i] if i in df else 0 for i, v in self.annotation.items()}
        return self.percolation_values

    def build_base_network(self, 
                           start:str = 'start', 
                           to: str='to',
                           weight: str='weight',
                           edge_type: str = 'layer'):
        """
        Constructs the initial NetworkX DiGraph from the loaded network dataframe and annotations.
        """
        print(f"{time.ctime()} - Creating network")
        self.G = nx.DiGraph()

        # Add nodes with attributes
        nodes = [(index, {"node_type": value, "percolation": self.percolation_values.get(index, 0)}) 
                 for index, value in self.annotation.items()]
        self.G.add_nodes_from(nodes)

        # Add edges with attributes
        # Assuming columns: start, to, weight, edge_type
        # itertuples yields: (Index, start, to, weight, edge_type, ...)
        # Based on original script: value[1]=start, value[2]=to, value[3]=weight, value[4]=edge_type
        edges = []
        for row in self.net_df.itertuples(index=False):
            u = getattr(row, start)
            v = getattr(row, to)
            w = getattr(row, weight)
            et = getattr(row, edge_type)
            edges.append((u, v, {'weight': w, 'edge_type': et}))
            
        self.G.add_edges_from(edges)
        
        # Remove isolates
        self.G.remove_nodes_from(list(nx.isolates(self.G)))
        
        return self.G

    def update_edge_capacities(self, method='multiplicative', alpha=0.5):
        """
        Updates edge capacities based on weights and node expression (percolation) values.
        """
        # Create a series for easy lookup, similar to original df usage in update_edge_capacities
        # self.percolation_values is a dict
        
        for u, v, d in self.G.edges(data=True):
            weight = d.get('weight', 0)
            expression_u = self.G.nodes[u].get('percolation', 0)
            expression_v = self.G.nodes[v].get('percolation', 0)

            if method == 'multiplicative':
                d['capacity'] = weight * (expression_u + expression_v) / 2
            elif method == 'min_expression':
                d['capacity'] = weight * min(expression_u, expression_v)
            elif method == 'max_expression':
                d['capacity'] = weight * max(expression_u, expression_v)
            elif method == 'weighted_average':
                d['capacity'] = weight * (alpha * expression_u + (1 - alpha) * expression_v)
            else:
                raise ValueError(f"Unknown method {method}")

    def identify_targets(self, mediators_list):
        """
        Identifies target genes using regulons and a list of mediators.
        Returns a list of target genes.
        """
        if self.regulons is None:
            return []
            
        # Filter columns in regulons that contain at least one value from mediators_list
        filtered_regulon = self.regulons.loc[:, self.regulons.apply(lambda col: col.isin(mediators_list).any())]
        target_genes = filtered_regulon.columns.tolist()
        return target_genes

    def prepare_flow_topology(self, ligands, target_genes):
        """
        Adds SuperSource, SuperSink, and dummy nodes to prepare for Flow analysis.
        
        Args:
            ligands (list): List of ligand gene names (sources).
            target_genes (list): List of target gene names (sinks).
            
        Returns:
            sources (list): List of source nodes (ligands + SuperSource).
            sinks (list): List of sink nodes (dummy nodes + SuperSink).
        """
        # Calculate mean capacity for dummy edges
        capacities = [d['capacity'] for u, v, d in self.G.edges(data=True) if 'capacity' in d]
        mean_capacity = np.mean(capacities) if capacities else 1.0

        # --- Setup Sinks ---
        sinks = list(set(target_genes))
        supersink = [('SuperSink', {'node_type': 'TargetGenes', 'percolation': 0})]
        
        dummy_sink_edges = []
        dummy_source_edges = []
        
        dummy_sinks = []
        dummy_sources = []

        for node in sinks:
            if node in self.G: # Only add if node exists in graph
                dummy_name = f"dummy_{node}"
                dummy_sinks.append((dummy_name, {'node_type': 'dummy', 'percolation': 1}))
                
                # Edge from Gene -> Dummy
                dummy_sink_edges.append((node, dummy_name, {'weight': 1, 'edge_type': 'sink', 'capacity': mean_capacity}))
        
        # Edges from Dummy -> SuperSink
        sink_edges = []
        for node in dummy_sinks:
            sink_edges.append((node[0], 'SuperSink', {'weight': 1, 'edge_type': 'sink', 'capacity': 999}))

        self.G.add_nodes_from(supersink)
        self.G.add_nodes_from(dummy_sinks)
        self.G.add_edges_from(dummy_sink_edges)
        self.G.add_edges_from(sink_edges)

        # --- Setup Sources ---
        supersource = [('SuperSource', {'node_type': 'Source', 'percolation': 0})]
        
        # Connect SuperSource to Receptors that are neighbors of Ligands
        # Note: The original script logic was:
        # Iterate ligands -> find neighbors -> if neighbor is receptor -> connect SuperSource to Receptor
        
        connected_receptors = set()
        for node in ligands:
            if node in self.G:
                for neighbor in self.G.neighbors(node):
                    if neighbor not in connected_receptors:
                        attrs = self.G.nodes[neighbor]
                        if attrs.get("node_type") == "receptor":
                            connected_receptors.add(neighbor)

        # Create dummy source nodes similar to dummy sinks
        dummy_source_nodes = []
        dummy_source_edges = []
        
        for node in connected_receptors:
            dummy_name = f"dummy_{node}"
            dummy_source_nodes.append((dummy_name, {'node_type': 'dummy', 'percolation': 1}))
            
            # Edge from Receptor -> Dummy Source
            dummy_source_edges.append((dummy_name, node, {'weight': 1, 'edge_type': 'source', 'capacity': mean_capacity}))
        
        # Edges from Dummy Source -> SuperSource
        source_edges = []
        for node in dummy_source_nodes:
            source_edges.append(('SuperSource', node[0], {'weight': 1, 'edge_type': 'source', 'capacity': 999}))

        self.G.add_nodes_from(supersource)
        self.G.add_nodes_from(dummy_source_nodes)
        self.G.add_edges_from(dummy_source_edges)
        self.G.add_edges_from(source_edges)
        
        # Define sources and sinks lists for the GA
        final_sources = ['SuperSource'] + [n[0] for n in dummy_source_nodes]
        final_sinks = [n[0] for n in dummy_sinks] + ['SuperSink']
        
        return final_sources, final_sinks

    def get_graph(self):
        return self.G
