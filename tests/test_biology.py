import pytest
import pandas as pd
import numpy as np
import networkx as nx
from unittest.mock import MagicMock, patch
from EvoFlow.biology import BioNetworkPreparer

def test_initialization():
    prep = BioNetworkPreparer()
    assert prep.G is None
    assert prep.annotation is None
    assert prep.adata is None

def test_load_data(mock_net_df, mock_annotation, mock_anndata):
    prep = BioNetworkPreparer()
    
    # Mocking pandas.read_csv and scanpy.read to return our fixtures
    with patch('pandas.read_csv') as mock_read_csv:
        # Side effect to handle multiple calls to read_csv (network, annotation, regulons)
        def side_effect(path, **kwargs):
            if 'network' in path:
                return mock_net_df
            elif 'annotation' in path:
                return mock_annotation
            elif 'regulons' in path:
                return pd.DataFrame()
            return pd.DataFrame()
            
        mock_read_csv.side_effect = side_effect
        
        with patch('scanpy.read') as mock_sc_read:
            mock_sc_read.return_value = mock_anndata
            
            prep.load_data('network.txt', 'annotation.txt', 'file.h5ad')
            
            assert prep.net_df is not None
            assert prep.annotation is not None
            assert prep.adata is not None

def test_percolation_logic(mock_annotation, mock_anndata):
    prep = BioNetworkPreparer()
    prep.annotation = mock_annotation
    prep.adata = mock_anndata
    
    # Manually setting up a scenario:
    # 3 "Type X" cells.
    # Gene A: Present in 1 (33%)
    # Gene B: Present in 2 (66%)
    
    # We need to manually filter adata first (mocking filter_expression_data logic)
    # Since our fixture mock_anndata is a Mock, we can't easily rely on real pandas filtering inside the class
    # So we'll mock the internal state of prep.adata after filtering.
    
    # Let's say filtered matrix has 2 cells
    # Cell 1: [1, 0, ...]
    # Cell 2: [1, 1, ...]
    # Sum:    [2, 1, ...]
    # Prop:   [1.0, 0.5] -> Normalized: (x - min)/(max - min)
    
    # Instead of complex matrix math on mocks, let's verify the logic flow using a simpler mocked scenario
    # within the class method if possible, or trust the integration test.
    
    # Actually, let's use the method directly but with a real (small) AnnData if possible, 
    # OR inject the dataframes.
    
    # Injecting prepared data into the method logic:
    # The method uses: sp.sparse.csr_matrix.todense(self.adata.X)
    
    prep.adata.X = sp.sparse.csr_matrix([
        [1, 1],
        [1, 0]
    ])
    prep.adata.var_names = ['A', 'B']
    
    # Logic in method:
    # df = [2, 1] (sum)
    # min=1, max=2
    # norm = (val - 1) / (2 - 1) => (val - 1)
    # A: 2 -> 1.0
    # B: 1 -> 0.0
    
    # Mock annotation to match keys
    prep.annotation = pd.Series(['type']*2, index=['A', 'B'])
    
    percolation = prep.compute_percolation()
    
    assert percolation['A'] == 1.0
    assert percolation['B'] == 0.0

def test_build_network(mock_net_df, mock_annotation):
    prep = BioNetworkPreparer()
    prep.net_df = mock_net_df
    prep.annotation = mock_annotation
    prep.percolation_values = {'A': 0.5, 'B': 0.5, 'C': 0.5, 'D': 0.5, 'E': 0.5}
    
    G = prep.build_base_network()
    
    assert isinstance(G, nx.DiGraph)
    assert 'A' in G.nodes
    assert 'B' in G.nodes
    # Check Edge A->B
    assert G.has_edge('A', 'B')
    assert G.edges['A', 'B']['weight'] == 1.0
    
    # Check attributes
    assert G.nodes['A']['node_type'] == 'ligand'
    assert G.nodes['A']['percolation'] == 0.5

def test_update_capacities_multiplicative():
    prep = BioNetworkPreparer()
    G = nx.DiGraph()
    G.add_node('A', percolation=0.5)
    G.add_node('B', percolation=1.0)
    G.add_edge('A', 'B', weight=10.0)
    prep.G = G
    
    prep.update_edge_capacities(method='multiplicative')
    
    # Cap = Weight * (ExpA + ExpB) / 2
    # Cap = 10 * (0.5 + 1.0) / 2 = 10 * 0.75 = 7.5
    
    assert G.edges['A', 'B']['capacity'] == 7.5

def test_prepare_flow_topology():
    prep = BioNetworkPreparer()
    G = nx.DiGraph()
    # A(Ligand) -> B(Receptor) -> C -> D(Target)
    G.add_nodes_from([
        ('A', {'node_type': 'ligand'}), 
        ('B', {'node_type': 'receptor'}),
        ('C', {'node_type': 'mediator'}),
        ('D', {'node_type': 'target'})
    ])
    G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D')])
    prep.G = G
    
    ligands = ['A']
    targets = ['D']
    
    sources, sinks = prep.prepare_flow_topology(ligands, targets)
    
    # Check SuperSource
    assert 'SuperSource' in G.nodes
    assert 'SuperSource' in sources
    
    # SuperSource should connect to B (Receptor neighbor of A)
    # A -> B. A is ligand. Neighbors of A is B. B is Receptor.
    # So SuperSource -> B
    assert G.has_edge('SuperSource', 'B')
    
    # Check SuperSink
    assert 'SuperSink' in G.nodes
    assert 'SuperSink' in sinks
    
    # Check Dummy nodes
    # D is target. Should have dummy_D.
    # D -> dummy_D -> SuperSink
    assert 'dummy_D' in G.nodes
    assert G.has_edge('D', 'dummy_D')
    assert G.has_edge('dummy_D', 'SuperSink')
