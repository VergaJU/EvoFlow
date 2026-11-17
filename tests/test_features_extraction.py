import networkx as nx

from EvoFlow.features_extraction import NetworkFeatures


def test_get_log_search_space_zero_b():
    G = nx.DiGraph()
    G.add_nodes_from(range(10))
    nf = NetworkFeatures(G=G, b=0)
    assert nf.get_log_search_space() == 0.0


def test_get_log_search_space_b_gt_n():
    G = nx.DiGraph()
    G.add_nodes_from(range(5))
    nf = NetworkFeatures(G=G, b=10)
    assert nf.get_log_search_space() == 0.0


def test_get_log_search_space_normal():
    G = nx.DiGraph()
    G.add_nodes_from(range(10))
    nf = NetworkFeatures(G=G, b=3)
    result = nf.get_log_search_space()
    assert result > 0
    assert abs(result - 2.079) < 0.01


def test_calculate_density_single_node():
    G = nx.DiGraph()
    G.add_node("A")
    nf = NetworkFeatures(G=G, b=1)
    assert nf.calculate_density() == 0.0


def test_calculate_density_normal():
    G = nx.DiGraph()
    G.add_nodes_from(range(4))
    G.add_edges_from([(0, 1), (1, 2), (2, 3)])
    nf = NetworkFeatures(G=G, b=1)
    assert nf.calculate_density() == 0.25


def test_clustering_coefficient_small():
    G = nx.DiGraph()
    G.add_nodes_from(range(3))
    nf = NetworkFeatures(G=G, b=1)
    assert nf.calculate_clustering_coefficient() == 0.0


def test_clustering_coefficient_normal():
    G = nx.complete_graph(5, create_using=nx.DiGraph)
    nf = NetworkFeatures(G=G, b=2)
    coeff = nf.calculate_clustering_coefficient()
    assert coeff > 0


def test_estimate_alpha_too_few_degrees():
    G = nx.DiGraph()
    G.add_nodes_from(range(5))
    G.add_edge(0, 1)
    nf = NetworkFeatures(G=G, b=1)
    assert nf.estimate_alpha() == 0


def test_estimate_alpha_normal(complex_graph):
    nf = NetworkFeatures(G=complex_graph, b=3)
    alpha = nf.estimate_alpha()
    if alpha != 0:
        assert 1.5 <= alpha <= 10


def test_extract_features_keys(complex_graph):
    nf = NetworkFeatures(G=complex_graph, b=3)
    features = nf.extract_features()
    assert isinstance(features, dict)
    assert set(features.keys()) == {
        "log_search_space",
        "alpha",
        "density",
        "clustering_coef",
    }
    for v in features.values():
        assert isinstance(v, list)
        assert len(v) == 1


def test_extract_features_integration(complex_graph):
    nf = NetworkFeatures(G=complex_graph, b=3)
    features = nf.extract_features()
    assert features["log_search_space"][0] > 0
    assert features["density"][0] > 0
