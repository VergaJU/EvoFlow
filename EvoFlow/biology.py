import numpy as np
import pandas as pd
import networkx as nx
import scanpy as sc
import scipy as sp
import time
from anndata import AnnData
from .preparer import prepare_flow_topology

# Allowed values for BioNetworkPreparer.compute_percolation(); the CLI imports
# these as argparse choices so engine and CLI cannot drift.
EXPRESSION_STATS = ("nonzero_ratio", "mean", "median", "p90", "max")
EXPRESSION_NORMS = ("none", "cpm", "log1p", "minmax")


def normalize_expression(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """
    Normalize a samples x genes expression matrix.

    Args:
        df: Dense numeric matrix with samples/cells as rows and genes as columns.
        method: One of EXPRESSION_NORMS.

    Returns:
        Normalized copy of df (non-negative values preserved).
    """
    if method not in EXPRESSION_NORMS:
        raise ValueError(
            f"expression_norm must be one of {EXPRESSION_NORMS}, got {method!r}"
        )
    if method == "none":
        return df
    df = df.clip(lower=0.0)
    if method == "cpm":
        row_sums = df.sum(axis=1).replace(0.0, np.nan)
        return df.div(row_sums, axis=0).fillna(0.0) * 1e6
    if method == "log1p":
        return pd.DataFrame(np.log1p(df), index=df.index, columns=df.columns)
    # minmax: scale each gene (column) across samples to [0, 1]
    col_min = df.min(axis=0)
    span = (df.max(axis=0) - col_min).replace(0.0, np.nan)
    return (df - col_min).div(span, axis=1).fillna(0.0)


def transpose_expression(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orient a genes-as-rows bulk expression table to samples x genes.

    If the first column is non-numeric it is taken as gene IDs (index) before
    transposing; all-numeric frames are transposed as-is.
    """
    first_col = df.columns[0]
    if not pd.api.types.is_numeric_dtype(df[first_col]):
        df = df.set_index(first_col)
    return df.T


class BioNetworkPreparer:
    """
    Handles biological data loading, network construction, and capacity calculation
    based on gene expression data.
    """

    def __init__(self, lock_targets: int = 0):
        if lock_targets not in (0, 1, 2, 3):
            raise ValueError(f"lock_targets must be 0, 1, 2, or 3, got {lock_targets}")
        self.G = None
        self.annotation = None
        self.expression = None
        self.regulons = None
        self.percolation_values = False
        # Column names for network and annotation files
        self.network_start_col = "start"
        self.network_to_col = "to"
        self.network_weight_col = "weight"
        self.network_edge_type_col = "layer"
        self.annotation_index_col = "gene"
        # Node type for target identification in graph-based pathway
        self.target_node_type = "transcription_factor"
        self.lock_targets = lock_targets

    def load_data(
        self,
        network_path,
        annotation_path,
        expression_path: str = None,
        regulons_path=None,
        annotation_index_col: str = "gene",
        network_start_col: str = "start",
        network_to_col: str = "to",
        network_weight_col: str = "weight",
        network_edge_type_col: str = "layer",
        expression_transpose: bool = False,
        **kwargs,
    ):
        """
        Loads the raw input files with flexible column name support.

        Args:
            network_path (str): Path to network TSV file.
            annotation_path (str): Path to annotation TSV file.
            expression_path (str, optional): Path to expression data file (.h5ad, .csv, or .tsv).
            regulons_path (str, optional): Path to regulons CSV file.
            annotation_index_col (str): Column name for gene annotation index. Default: "gene".
            network_start_col (str): Network file column for source nodes. Default: "start".
            network_to_col (str): Network file column for target nodes. Default: "to".
            network_weight_col (str): Network file column for edge weights. Default: "weight".
            network_edge_type_col (str): Network file column for edge types. Default: "layer".
            expression_transpose (bool): Bulk csv/tsv files have genes as rows (first
                column gene IDs); set True to orient them to samples x genes.
                Not supported for h5ad. Default: False.
            **kwargs: Additional keyword arguments for extensibility.
        """
        print(f"{time.ctime()} - Reading files")
        # Store column names for later use
        self.annotation_index_col = annotation_index_col
        self.network_start_col = network_start_col
        self.network_to_col = network_to_col
        self.network_weight_col = network_weight_col
        self.network_edge_type_col = network_edge_type_col

        # Load Network
        self.net_df = pd.read_csv(network_path, sep="\t")
        # Handle "from" column rename to "start"
        if "from" in self.net_df.columns and network_start_col == "start":
            self.net_df = self.net_df.rename(columns={"from": "start"})

        # Load Annotation
        self.annotation = pd.read_csv(
            annotation_path, sep="\t", index_col=annotation_index_col
        ).squeeze()

        if expression_path:
            if expression_transpose and expression_path.endswith(".h5ad"):
                raise ValueError(
                    "expression_transpose is only supported for csv/tsv expression "
                    "files; h5ad is already samples x genes"
                )
            try:
                if expression_path.endswith(".h5ad"):
                    self.expression = sc.read(expression_path)
                elif expression_path.endswith((".csv", ".tsv")):
                    df = pd.read_csv(
                        expression_path,
                        sep="\t" if expression_path.endswith(".tsv") else ",",
                    )
                    if expression_transpose:
                        df = transpose_expression(df)
                    self.expression = df
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

    def compute_percolation(self, stat: str = "nonzero_ratio", norm: str = "none"):
        """
        Computes per-gene percolation values used to weight edge capacities.

        The expression matrix is reduced to one non-negative value per gene
        (columns are genes; samples/cells are rows), then scaled by its maximum
        so percolation stays in (0, 1].

        Args:
            stat: Gene statistic, one of EXPRESSION_STATS. "nonzero_ratio"
                (default) counts expressing samples and ignores `norm`.
            norm: Normalization applied before the statistic for stats other
                than "nonzero_ratio", one of EXPRESSION_NORMS.

        Returns:
            dict: annotation gene -> percolation value (0.0 for genes absent
            from the expression matrix). `percolation_values` is set to False
            when no expression data was loaded.
        """
        if stat not in EXPRESSION_STATS:
            raise ValueError(
                f"expression_stat must be one of {EXPRESSION_STATS}, got {stat!r}"
            )
        if norm not in EXPRESSION_NORMS:
            raise ValueError(
                f"expression_norm must be one of {EXPRESSION_NORMS}, got {norm!r}"
            )
        if self.expression is None:
            self.percolation_values = False
            return self.percolation_values

        # Export count matrix to a samples x genes DataFrame
        if isinstance(self.expression, AnnData):
            df = pd.DataFrame(sp.sparse.csr_matrix.todense(self.expression.X))
            df.columns = self.expression.var_names
        else:
            df = self.expression.copy()
        df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

        if stat == "nonzero_ratio":
            # Raw nonzero mask; normalization deliberately not applied
            # (min-max would shift nonzero entries to zero).
            values = df.astype("bool").sum()
        else:
            df = normalize_expression(df, norm)
            if stat == "mean":
                values = df.mean()
            elif stat == "median":
                values = df.median()
            elif stat == "p90":
                values = df.quantile(0.9)
            else:  # max
                values = df.max()

        if (
            len(values) > 0
            and len(self.annotation) > 0
            and not any(g in values.index for g in self.annotation.index)
        ):
            print(
                "Warning: expression matrix shares no genes with the annotation; "
                "percolation set to 0 everywhere. If the file has genes as rows, "
                "pass expression_transpose=True (--expression_transpose)."
            )

        values = values.clip(lower=0.0)
        max_value = values.max() if len(values) else 0.0
        if max_value > 0:
            values = values / max_value
        else:
            # Handle edge case where the expression matrix is entirely zeros
            values = values * 0.0

        # Create dictionary with per-gene percolation values. Cast to Python
        # float: np.float32 (AnnData's default dtype) is not JSON-serializable
        # and would break the cyjs/graphml savers downstream.
        self.percolation_values = {
            i: float(values.loc[i]) if i in values.index else 0.0
            for i in self.annotation.index
        }
        return self.percolation_values

    def build_base_network(
        self,
        start: str = None,
        to: str = None,
        weight: str = None,
        edge_type: str = None,
    ):
        """
        Constructs the initial NetworkX DiGraph from the loaded network dataframe and annotations.

        Args:
            start (str, optional): Column name for source nodes. Uses stored value if not provided.
            to (str, optional): Column name for target nodes. Uses stored value if not provided.
            weight (str, optional): Column name for edge weights. Uses stored value if not provided.
            edge_type (str, optional): Column name for edge types. Uses stored value if not provided.
        """
        # Use provided values or fall back to stored instance variables
        start = start or self.network_start_col
        to = to or self.network_to_col
        weight = weight or self.network_weight_col
        edge_type = edge_type or self.network_edge_type_col

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

    def identify_targets(
        self, mediators_list, target_node_type: str = "transcription_factor", **kwargs
    ):
        """
        Identifies target genes based on available data sources.

        If regulons are available: uses regulon-based target identification.
        If regulons are not available: finds upstream nodes with specified node_type
        for each mediator in the graph.

        Args:
            mediators_list (list): List of mediator gene names.
            target_node_type (str): Node type to search for in graph-based pathway.
                                   Default: "transcription_factor".
            **kwargs: Additional keyword arguments for extensibility.

        Returns:
            list: List of target genes.
        """
        if len(mediators_list) == 0:
            return []

        # Store the target node type for later use
        self.target_node_type = target_node_type

        if self.regulons is not None:
            # Regulons-based pathway
            # Filter columns in regulons that contain at least one value from mediators_list
            filtered_regulon = self.regulons.loc[
                :, self.regulons.apply(lambda col: col.isin(mediators_list).any())
            ]
            target_genes = filtered_regulon.columns.tolist()
            return target_genes
        else:
            # Graph-based pathway: find upstream nodes of specified type
            target_genes = set()

            # For each mediator, find upstream nodes with matching node_type
            for mediator in mediators_list:
                # If mediator is in the graph
                if mediator in self.G:
                    # Check if the mediator itself has the target node type
                    if self.G.nodes[mediator].get("node_type") == target_node_type:
                        target_genes.add(mediator)

                    # Get all predecessors (upstream nodes) with matching node_type
                    for predecessor in self.G.predecessors(mediator):
                        if (
                            self.G.nodes[predecessor].get("node_type")
                            == target_node_type
                        ):
                            target_genes.add(predecessor)

            return list(target_genes)

    def prepare_bioflow_topology(self, ligands=None, target_genes=None, receptors=None):
        """
        Biological wrapper for flow topology preparation: derives sources/sinks from ligands/targets, then calls general function.

        Args:
            ligands (list, optional): List of ligand gene names. Use this to derive receptors from ligand neighbors.
            target_genes (list, optional): List of target gene names.
            receptors (list, optional): List of receptor gene names to use as source nodes directly.

        Returns:
            tuple: (sources, sinks) as returned by the general prepare_flo_topology.

        Raises:
            ValueError: If both ligands and receptors are provided, or if neither are provided.
        """
        # Validate that at least one source specification is provided
        if ligands is None and receptors is None:
            raise ValueError("Either ligands or receptors must be provided")
        if ligands is not None and receptors is not None:
            raise ValueError(
                "Specify only one of ligands or receptors, not both. "
                "Use ligands to auto-derive receptors, or provide receptors directly."
            )

        # Derive sink_nodes (targets)
        sink_nodes = list(set(target_genes)) if target_genes else []

        # Derive source_nodes based on what was provided
        if receptors is not None:
            # Receptors pathway: filter provided receptors against graph nodes
            source_nodes = [r for r in receptors if r in self.G]
        else:
            # Ligands pathway: derive connected receptors from ligand neighbors
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
        return prepare_flow_topology(
            self.G, source_nodes, sink_nodes, self.lock_targets
        )

    def get_graph(self):
        return self.G

    def prepare_network(
        self,
        network_path: str,
        annotation_path: str,
        expression_path: str,
        ligands: list = None,
        mediators_list: list = None,
        label: str = None,
        celltype: str = None,
        regulons_path: str = None,
        targets: list = None,
        receptors: list = None,
        lock_targets: int = 0,
        expression_stat: str = "nonzero_ratio",
        expression_norm: str = "none",
        expression_transpose: bool = False,
        **kwargs,
    ) -> tuple:
        """
        Orchestrates the complete network preparation workflow: load data, filter expression,
        compute percolation, build network, update capacities, identify targets, and prepare topology.

        Args:
            network_path (str): Path to network file.
            annotation_path (str): Path to annotation file.
            expression_path (str): Path to expression data file (.h5ad, .csv, or .tsv).
            ligands (list, optional): List of ligand gene names. Use to auto-derive receptors from neighbors.
            mediators_list (list, optional): List of mediator/receptor gene names.
            label (str, optional): Column name in AnnData obs for cell type filtering (h5ad only).
            celltype (str, optional): Cell type to filter by (h5ad only).
            regulons_path (str, optional): Path to regulons file for target identification.
            targets (list, optional): List of target genes (used if regulons_path is None).
            receptors (list, optional): List of receptor gene names to use as source nodes directly.
            lock_targets (int): Bitmask controlling which original nodes are locked into sources/sinks.
                0 (default): No locking.
                1: Original source nodes are locked into final_sources.
                2: Original sink nodes are locked into final_sinks.
                3: Both source and sink nodes are locked.
            expression_stat (str): Per-gene statistic for percolation, one of
                EXPRESSION_STATS (default: "nonzero_ratio").
            expression_norm (str): Normalization applied before the statistic
                (ignored for "nonzero_ratio"), one of EXPRESSION_NORMS (default: "none").
            expression_transpose (bool): Orient genes-as-rows csv/tsv bulk files to
                samples x genes. Not supported for h5ad (default: False).
            **kwargs: Additional parameters for column name customization:
                - annotation_index_col: Custom annotation index column name (default: "gene")
                - network_start_col: Custom network start column name (default: "start")
                - network_to_col: Custom network target column name (default: "to")
                - network_weight_col: Custom network weight column name (default: "weight")
                - network_edge_type_col: Custom network edge type column name (default: "layer")

        Returns:
            tuple: (graph, sources, sinks) where graph is the prepared NetworkX DiGraph
                   and sources/sinks are lists for flow optimization.

        Raises:
            ValueError: If ligands and receptors are both missing or both provided, if both
                regulons_path and targets are None, if expression_stat/expression_norm are
                invalid, or if only one of label/celltype is given.
        """
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

        # Input validation - check for ligands/receptors
        has_ligands = ligands is not None and len(ligands) > 0
        has_receptors = receptors is not None and len(receptors) > 0

        if not has_ligands and not has_receptors:
            # Provide specific error message based on what was provided
            if ligands is not None and len(ligands) == 0:
                raise ValueError("ligands list cannot be empty")
            else:
                raise ValueError("Either ligands or receptors must be provided")
        if has_ligands and has_receptors:
            raise ValueError(
                "Specify only one of ligands or receptors, not both. "
                "Use ligands to auto-derive receptors, or provide receptors directly."
            )

        # Validate mediators_list and targets
        if mediators_list and len(mediators_list) == 0:
            raise ValueError("mediators_list cannot be empty when provided")
        if regulons_path and not mediators_list:
            raise ValueError("mediators_list cannot be empty when using regulons")
        if regulons_path is None and targets is None:
            raise ValueError("Either regulons_path or targets must be provided")

        # Load data with custom column names if provided
        self.load_data(
            network_path,
            annotation_path,
            expression_path,
            regulons_path,
            expression_transpose=expression_transpose,
            **kwargs,
        )

        # Filter expression data (AnnData/cell metadata only)
        if isinstance(self.expression, AnnData) and label is not None:
            self.filter_expression_data(label, celltype)
        elif label is not None:
            print(
                "Warning: label/celltype filtering only applies to h5ad expression "
                "data; ignoring."
            )

        # Compute percolation
        self.compute_percolation(stat=expression_stat, norm=expression_norm)

        # Build base network (column names are now stored in instance)
        self.build_base_network()

        # Update capacities
        self.update_edge_capacities(method="multiplicative")

        # Identify targets
        if regulons_path:
            target_genes = self.identify_targets(mediators_list)
        elif targets:
            target_genes = targets

        # Prepare topology and get sources/sinks
        self.lock_targets = lock_targets
        sources, sinks = self.prepare_bioflow_topology(
            ligands=ligands, target_genes=target_genes, receptors=receptors
        )

        # Return graph and topology
        return self.G, sources, sinks
