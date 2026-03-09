import pytest
import pandas as pd
import networkx as nx
import scipy.sparse as sp
import anndata as ad

# --- Fixtures for Biology Tests ---


@pytest.fixture
def mock_net_df():
    """Returns a simple network dataframe"""
    data = {
        "start": ["A", "B", "C", "D"],
        "to": ["B", "C", "D", "E"],
        "weight": [1.0, 0.8, 0.5, 0.9],
        "layer": ["activation", "inhibition", "activation", "activation"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_annotation():
    """Returns a simple annotation series"""
    index = ["A", "B", "C", "D", "E"]
    data = ["ligand", "receptor", "mediator", "mediator", "target"]
    return pd.Series(data, index=index, name="node_type")


@pytest.fixture
def mock_anndata():
    """Returns a real AnnData object with sparse matrix X and obs"""
    # 5 genes (cols), 4 cells (rows)
    # Genes: A, B, C, D, E

    # Create sparse matrix (4 cells x 5 genes)
    X = sp.csr_matrix(
        [
            [1, 0, 0, 0, 1],  # Cell 1 (Type X)
            [1, 0, 1, 0, 1],  # Cell 2 (Type X)
            [1, 0, 1, 1, 1],  # Cell 3 (Type Y)
            [1, 0, 0, 0, 0],  # Cell 4 (Type X)
        ]
    )

    var_names = ["A", "B", "C", "D", "E"]
    obs_data = {"cell_type_col": ["Type X", "Type X", "Type Y", "Type X"]}

    return ad.AnnData(
        X=X, var=pd.DataFrame(index=var_names), obs=pd.DataFrame(obs_data)
    )


@pytest.fixture
def mock_regulons():
    """Returns a mock regulons dataframe"""
    data = {
        "TF": ["A", "B"],
        "Target1": ["E", "C"],
        "Target2": ["D", "F"],  # F doesn't exist in net
    }
    return pd.DataFrame(data)


# --- Fixtures for Core Tests ---


@pytest.fixture
def simple_graph():
    """
    Creates a simple Directed Graph for flow testing.
    Structure: SuperSource -> A -> B -> SuperSink
    """
    G = nx.DiGraph()
    G.add_node("SuperSource")
    G.add_node("A", percolation=1.0)
    G.add_node("B", percolation=0.5)
    G.add_node("SuperSink")

    # Path: S -> A -> B -> T
    # Capacities needed for flow
    G.add_edge("SuperSource", "A", capacity=10.0, weight=1.0)
    G.add_edge("A", "B", capacity=5.0, weight=1.0)
    G.add_edge("B", "SuperSink", capacity=10.0, weight=1.0)

    return G


@pytest.fixture
def branching_graph():
    """
    Structure:
       /-> A ->\
    S           T
       \-> B ->/
    """
    G = nx.DiGraph()
    G.add_nodes_from(["SuperSource", "A", "B", "SuperSink"])
    G.add_edge("SuperSource", "A", capacity=2.0)
    G.add_edge("SuperSource", "B", capacity=3.0)
    G.add_edge("A", "SuperSink", capacity=2.0)
    G.add_edge("B", "SuperSink", capacity=3.0)
    return G
