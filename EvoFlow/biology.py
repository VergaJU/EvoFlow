import pandas as pd
import networkx as nx
import scanpy as sc
import scipy as sp
import time
from anndata import AnnData
from .preparer import prepare_flow_topology


class BioNetworkPreparer:
    """
    Handles biological data loading, network construction, and capacity calculation
    based on gene expression data.
    """

    def __init__(self):
        self.G = None
        self.annotation = None
        self.expression = None
        self.regulons = None

    def load_data(
        self,
        network_path,
        annotation_path,
        expression_path: str = None,
        regulons_path=None,
    ):
        """
        Loads the raw input files.
        """
        print(f"{time.ctime()} - Reading files")
        # Load Network
        self.net_df = pd.read_csv(network_path, sep="\t")
        if "from" in self.net_df.columns:
            self.net_df = self.net_df.rename(columns={"from": "start"})

        # Load Annotation
        self.annotation = pd.read_csv(
            annotation_path, sep="\t", index_col="gene"
        ).squeeze()

        if expression_path:
            try:
                if expression_path.endswith(".h5ad"):
                    self.expression = sc.read(expression_path)
                elif expression_path.endswith((".csv", ".tsv")):
                    self.expression = pd.read_csv(
                        expression_path,
                        sep="\t" if expression_path.endswith(".tsv") else ",",
                    )
                else:
                    raise ValueError(
                        "Input must be anndata or tsv,csv expression matrices"
                    )
            except Exception as e:
                raise ValueError(
                    "Input must be anndata or tsv,csv expression matrices"
                ) from e

        # Load Regulons (Optional)
        if regulons_path:
            self.regulons = pd.read_csv(regulons_path)
            if 0 in self.regulons.index:
                self.regulons = self.regulons.drop(0)

    def filter_expression_data(self, label_col, cell_type):
        """
        Filters the AnnData object for a specific cell type.
        """
        self.expression = self.expression[self.expression.obs[label_col] == cell_type]

    def compute_percolation(self):
        """
        Computes the proportion of gene-expressing cells to be used as percolation values and compute capacities of the edges.
        """
        # Export count matrix and transform it to a Series with count of gene-expressing cells
        if type(self.expression) == AnnData:
            df = pd.DataFrame(sp.sparse.csr_matrix.todense(self.expression.X))
            df.columns = self.expression.var_names
        else:
            df = self.expression.copy()

        # Count the number of cells expressing each gene
        df = df.astype("bool").sum()

        # New Normalization Logic: Divide by the maximum count
        max_count = df.max()

        if max_count != 0:
            df = df / max_count
        else:
            df[:] = (
                0.0  # Handle edge case where the expression matrix is entirely zeros
            )

        # Create dictionary with proportion of gene-expressing cells
        self.percolation_values = {
            i: df.loc[i] if i in df.index else 0.0 for i in self.annotation.index
        }
        return self.percolation_values

    def build_base_network(
        self,
        start: str = "start",
        to: str = "to",
        weight: str = "weight",
        edge_type: str = "layer",
    ):
        """
        Constructs the initial NetworkX DiGraph from the loaded network dataframe and annotations.
        """
        print(f"{time.ctime()} - Creating network")
        self.G = nx.DiGraph()

        if self.percolation_values:
            # Add nodes with percolation if expression is provided
            nodes = [
                (
                    index,
                    {
                        "node_type": value,
                        "percolation": self.percolation_values.get(index, 0),
                    },
                )
                for index, value in self.annotation.items()
            ]

        else:
            # add nodes without percolation equal to 1 if expression is not provided
            nodes = [
                (
                    index,
                    {
                        "node_type": value,
                        "percolation": 1,
                    },
                )
                for index, value in self.annotation.items()
            ]

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
            edges.append((u, v, {"weight": w, "edge_type": et}))

        self.G.add_edges_from(edges)

        # Remove isolates
        self.G.remove_nodes_from(list(nx.isolates(self.G)))

        return self.G

    def update_edge_capacities(self, method="multiplicative", alpha=0.5):
        """
        Updates edge capacities based on weights and node expression (percolation) values.
        """
        # Create a series for easy lookup, similar to original df usage in update_edge_capacities
        # self.percolation_values is a dict

        for u, v, d in self.G.edges(data=True):
            weight = d.get("weight", 0)
            expression_u = self.G.nodes[u].get("percolation", 0)
            expression_v = self.G.nodes[v].get("percolation", 0)

            if method == "multiplicative":
                d["capacity"] = weight * (expression_u + expression_v) / 2
            elif method == "min_expression":
                d["capacity"] = weight * min(expression_u, expression_v)
            elif method == "max_expression":
                d["capacity"] = weight * max(expression_u, expression_v)
            elif method == "weighted_average":
                d["capacity"] = weight * (
                    alpha * expression_u + (1 - alpha) * expression_v
                )
            else:
                raise ValueError(f"Unknown method {method}")

    def identify_targets(self, mediators_list):
        """
        Identifies target genes using regulons and a list of mediators.
        Returns a list of target genes.
        """
        if self.regulons is None:
            return []
        elif len(mediators_list) == 0:
            raise ValueError(
                "Regulons based target idenfication requires at leas some receptors or mediators"
            )

        # Filter columns in regulons that contain at least one value from mediators_list
        filtered_regulon = self.regulons.loc[
            :, self.regulons.apply(lambda col: col.isin(mediators_list).any())
        ]
        target_genes = filtered_regulon.columns.tolist()
        return target_genes

    def prepare_bioflow_topology(self, ligands, target_genes):
        """
        Biological wrapper for flow topology preparation: derives sources/sinks from ligands/targets, then calls general function.

        Args:
            ligands (list): List of ligand gene names.
            target_genes (list): List of target gene names.

        Returns:
            tuple: (sources, sinks) as returned by the general prepare_flo_topology.
        """
        # Derive sink_nodes (targets)
        sink_nodes = list(set(target_genes))

        # Derive source_nodes (receptors neighboring ligands)
        connected_receptors = set()
        for node in ligands:
            if node in self.G:
                for neighbor in self.G.neighbors(node):
                    if (
                        neighbor not in connected_receptors
                        and self.G.nodes[neighbor].get("node_type") == "receptor"
                    ):
                        connected_receptors.add(neighbor)
        source_nodes = list(connected_receptors)

        # Call general function
        return prepare_flow_topology(self.G, source_nodes, sink_nodes)

    def get_graph(self):
        return self.G

    def prepare_network(
        self,
        network_path: str,
        annotation_path: str,
        expression_path: str,
        ligands: list,
        mediators_list: list,
        label: str = None,
        celltype: str = None,
        regulons_path: str = None,
        targets: list = None,
    ) -> tuple:
        """
        Orchestrates the complete network preparation workflow: load data, filter expression,
        compute percolation, build network, update capacities, identify targets, and prepare topology.

        Args:
            network_path (str): Path to network file.
            annotation_path (str): Path to annotation file.
            expression_path (str): Path to expression data file (.h5ad, .csv, or .tsv).
            ligands (list): List of ligand gene names (source nodes).
            mediators_list (list): List of mediator/receptor gene names.
            label (str, optional): Column name in AnnData obs for cell type filtering.
            celltype (str, optional): Cell type to filter by.
            regulons_path (str, optional): Path to regulons file for target identification.
            targets (list, optional): List of target genes (used if regulons_path is None).

        Returns:
            tuple: (graph, sources, sinks) where graph is the prepared NetworkX DiGraph
                   and sources/sinks are lists for flow optimization.

        Raises:
            ValueError: If ligands or mediators_list are empty, or if both regulons_path and targets are None.
        """
        # Input validation
        if not ligands:
            raise ValueError("ligands list cannot be empty")
        if regulons_path and not mediators_list:
            raise ValueError("mediators_list cannot be empty when using regulons")
        if regulons_path is None and targets is None:
            raise ValueError("Either regulons_path or targets must be provided")

        # Load data
        self.load_data(network_path, annotation_path, expression_path, regulons_path)

        # Filter expression data if AnnData
        if type(self.expression) == AnnData:
            self.filter_expression_data(label, celltype)

        # Compute percolation
        self.compute_percolation()

        # Build base network
        self.build_base_network()

        # Update capacities
        self.update_edge_capacities(method="multiplicative")

        # Identify targets
        if regulons_path:
            target_genes = self.identify_targets(mediators_list)
        elif targets:
            target_genes = targets

        # Prepare topology and get sources/sinks
        sources, sinks = self.prepare_bioflow_topology(ligands, target_genes)

        # Return graph and topology
        return self.G, sources, sinks
