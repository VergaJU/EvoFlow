from unittest.mock import patch

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from EvoFlow.biology import (
    BioNetworkPreparer,
    normalize_expression,
    transpose_expression,
)


def test_initialization():
    prep = BioNetworkPreparer()
    assert prep.G is None
    assert prep.annotation is None
    assert prep.expression is None
    assert prep.lock_targets == 0

    prep2 = BioNetworkPreparer(lock_targets=3)
    assert prep2.lock_targets == 3

    with pytest.raises(ValueError):
        BioNetworkPreparer(lock_targets=5)


def test_load_data(mock_net_df, mock_annotation, mock_anndata):
    prep = BioNetworkPreparer()

    # Mocking pandas.read_csv and scanpy.read to return our fixtures
    with patch("pandas.read_csv") as mock_read_csv:
        # Side effect to handle multiple calls to read_csv (network, annotation, regulons)
        def side_effect(path, **kwargs):
            if "network" in path:
                return mock_net_df
            elif "annotation" in path:
                return mock_annotation
            elif "regulons" in path:
                return pd.DataFrame()
            return pd.DataFrame()

        mock_read_csv.side_effect = side_effect

        with patch("scanpy.read") as mock_sc_read:
            mock_sc_read.return_value = mock_anndata

            prep.load_data("network.txt", "annotation.txt", "file.h5ad")

            assert prep.net_df is not None
            assert prep.annotation is not None
            assert prep.expression is not None


def test_percolation_logic(mock_annotation, mock_anndata):
    prep = BioNetworkPreparer()
    prep.expression = mock_anndata

    # 4 cells x 5 genes (A..E); nonzero counts per gene:
    #   A=4, B=0, C=2, D=1, E=3
    # Default stat "nonzero_ratio" divides by the max count (4):
    #   A=1.0, B=0.0, C=0.5, D=0.25, E=0.75
    prep.annotation = pd.Series(
        ["type"] * len(prep.expression.var_names), index=prep.expression.var_names
    )

    percolation = prep.compute_percolation()

    assert percolation["A"] == 1.0
    assert percolation["B"] == 0.0
    assert percolation["C"] == 0.5
    assert percolation["D"] == 0.25
    assert percolation["E"] == 0.75


def test_build_network(mock_net_df, mock_annotation):
    prep = BioNetworkPreparer()
    prep.net_df = mock_net_df
    prep.annotation = mock_annotation
    prep.percolation_values = {"A": 0.5, "B": 0.5, "C": 0.5, "D": 0.5, "E": 0.5}

    G = prep.build_base_network()

    assert isinstance(G, nx.DiGraph)
    assert "A" in G.nodes
    assert "B" in G.nodes
    # Check Edge A->B
    assert G.has_edge("A", "B")
    assert G.edges["A", "B"]["weight"] == 1.0

    # Check attributes
    assert G.nodes["A"]["node_type"] == "ligand"
    assert G.nodes["A"]["percolation"] == 0.5


def test_update_capacities_multiplicative():
    prep = BioNetworkPreparer()
    G = nx.DiGraph()
    G.add_node("A", percolation=0.5)
    G.add_node("B", percolation=1.0)
    G.add_edge("A", "B", weight=10.0)
    prep.G = G

    prep.update_edge_capacities(method="multiplicative")

    # Cap = Weight * (ExpA + ExpB) / 2
    # Cap = 10 * (0.5 + 1.0) / 2 = 10 * 0.75 = 7.5

    assert G.edges["A", "B"]["capacity"] == 7.5


def test_prepare_bioflow_topology():
    prep = BioNetworkPreparer()
    G = nx.DiGraph()
    # A(Ligand) -> B(Receptor) -> C -> D(Target)
    G.add_nodes_from(
        [
            ("A", {"node_type": "ligand"}),
            ("B", {"node_type": "receptor"}),
            ("C", {"node_type": "mediator"}),
            ("D", {"node_type": "target"}),
        ]
    )
    G.add_edges_from([("A", "B"), ("B", "C"), ("C", "D")])
    prep.G = G

    ligands = ["A"]
    targets = ["D"]

    sources, sinks = prep.prepare_bioflow_topology(ligands, targets)

    # Check SuperSource
    assert "SuperSource" in G.nodes
    assert "SuperSource" in sources

    # SuperSource should connect to B (Receptor neighbor of A)
    # A -> B. A is ligand. Neighbors of A is B. B is Receptor.
    # So SuperSource -> B
    assert G.has_edge("dummy_B", "B")
    assert G.has_edge("SuperSource", "dummy_B")

    # Check SuperSink
    assert "SuperSink" in G.nodes
    assert "SuperSink" in sinks

    # Check Dummy nodes
    # D is target. Should have dummy_D.
    # D -> dummy_D -> SuperSink
    assert "dummy_D" in G.nodes
    assert G.has_edge("D", "dummy_D")
    assert G.has_edge("dummy_D", "SuperSink")


def test_prepare_flow_topology():
    from EvoFlow.preparer import prepare_flow_topology

    G = nx.DiGraph()
    # Add some nodes and edges with capacities
    G.add_nodes_from(["A", "B", "C", "D"])
    G.add_edges_from(
        [
            ("A", "B", {"capacity": 1.0}),
            ("B", "C", {"capacity": 2.0}),
            ("C", "D", {"capacity": 1.5}),
        ]
    )

    source_nodes = ["A"]
    sink_nodes = ["D"]

    sources, sinks = prepare_flow_topology(G, source_nodes, sink_nodes)

    # Check SuperSource and SuperSink added
    assert "SuperSource" in G.nodes
    assert "SuperSink" in G.nodes

    # Check dummy nodes
    assert "dummy_A" in G.nodes
    assert "dummy_D" in G.nodes

    # Check edges
    assert G.has_edge("SuperSource", "dummy_A")
    assert G.has_edge("dummy_A", "A")
    assert G.has_edge("D", "dummy_D")
    assert G.has_edge("dummy_D", "SuperSink")

    # Check sources and sinks lists
    assert sources == ["SuperSource", "dummy_A"]
    assert sinks == ["dummy_D", "SuperSink"]


def test_prepare_flow_topology_lock_0():
    from EvoFlow.preparer import prepare_flow_topology

    G = nx.DiGraph()
    G.add_nodes_from(["A", "B", "C", "D"])
    G.add_edges_from(
        [
            ("A", "B", {"capacity": 1.0}),
            ("B", "C", {"capacity": 2.0}),
            ("C", "D", {"capacity": 1.5}),
        ]
    )

    source_nodes = ["A"]
    sink_nodes = ["D"]

    sources, sinks = prepare_flow_topology(G, source_nodes, sink_nodes, lock_targets=0)

    assert "SuperSource" in sources
    assert "dummy_A" in sources
    assert "A" not in sources
    assert "D" not in sinks
    assert "dummy_D" in sinks
    assert "SuperSink" in sinks


def test_prepare_flow_topology_lock_1():
    from EvoFlow.preparer import prepare_flow_topology

    G = nx.DiGraph()
    G.add_nodes_from(["A", "B", "C", "D"])
    G.add_edges_from(
        [
            ("A", "B", {"capacity": 1.0}),
            ("B", "C", {"capacity": 2.0}),
            ("C", "D", {"capacity": 1.5}),
        ]
    )

    source_nodes = ["A"]
    sink_nodes = ["D"]

    sources, sinks = prepare_flow_topology(G, source_nodes, sink_nodes, lock_targets=1)

    assert "SuperSource" in sources
    assert "dummy_A" in sources
    assert "A" in sources
    assert "D" not in sinks
    assert "dummy_D" in sinks
    assert "SuperSink" in sinks


def test_prepare_flow_topology_lock_2():
    from EvoFlow.preparer import prepare_flow_topology

    G = nx.DiGraph()
    G.add_nodes_from(["A", "B", "C", "D"])
    G.add_edges_from(
        [
            ("A", "B", {"capacity": 1.0}),
            ("B", "C", {"capacity": 2.0}),
            ("C", "D", {"capacity": 1.5}),
        ]
    )

    source_nodes = ["A"]
    sink_nodes = ["D"]

    sources, sinks = prepare_flow_topology(G, source_nodes, sink_nodes, lock_targets=2)

    assert "SuperSource" in sources
    assert "dummy_A" in sources
    assert "A" not in sources
    assert "D" in sinks
    assert "dummy_D" in sinks
    assert "SuperSink" in sinks


def test_prepare_flow_topology_lock_3():
    from EvoFlow.preparer import prepare_flow_topology

    G = nx.DiGraph()
    G.add_nodes_from(["A", "B", "C", "D"])
    G.add_edges_from(
        [
            ("A", "B", {"capacity": 1.0}),
            ("B", "C", {"capacity": 2.0}),
            ("C", "D", {"capacity": 1.5}),
        ]
    )

    source_nodes = ["A"]
    sink_nodes = ["D"]

    sources, sinks = prepare_flow_topology(G, source_nodes, sink_nodes, lock_targets=3)

    assert "SuperSource" in sources
    assert "dummy_A" in sources
    assert "A" in sources
    assert "D" in sinks
    assert "dummy_D" in sinks
    assert "SuperSink" in sinks


def test_prepare_flow_topology_lock_invalid():
    from EvoFlow.preparer import prepare_flow_topology

    G = nx.DiGraph()
    G.add_nodes_from(["A", "B"])
    G.add_edge("A", "B", capacity=1.0)

    with pytest.raises(ValueError, match="lock_targets must be 0, 1, 2, or 3"):
        prepare_flow_topology(G, ["A"], ["B"], lock_targets=4)

    with pytest.raises(ValueError, match="lock_targets must be 0, 1, 2, or 3"):
        prepare_flow_topology(G, ["A"], ["B"], lock_targets=-1)


def test_prepare_bioflow_topology_lock_targets_propagation():
    prep = BioNetworkPreparer(lock_targets=1)
    G = nx.DiGraph()
    # A(Ligand) -> B(Receptor) -> C -> D(Target)
    G.add_nodes_from(
        [
            ("A", {"node_type": "ligand"}),
            ("B", {"node_type": "receptor"}),
            ("C", {"node_type": "mediator"}),
            ("D", {"node_type": "target"}),
        ]
    )
    G.add_edges_from([("A", "B"), ("B", "C"), ("C", "D")])
    prep.G = G

    sources, sinks = prep.prepare_bioflow_topology(ligands=["A"], target_genes=["D"])

    # lock_targets=1: original source_nodes (B, the receptor derived from A) should be in sources
    assert "B" in sources
    # lock_targets=1: original sink_nodes (D) should NOT be in sinks
    assert "D" not in sinks


def test_prepare_bioflow_topology_lock_targets_full():
    prep = BioNetworkPreparer(lock_targets=3)
    G = nx.DiGraph()
    # A(Ligand) -> B(Receptor) -> C -> D(Target)
    G.add_nodes_from(
        [
            ("A", {"node_type": "ligand"}),
            ("B", {"node_type": "receptor"}),
            ("C", {"node_type": "mediator"}),
            ("D", {"node_type": "target"}),
        ]
    )
    G.add_edges_from([("A", "B"), ("B", "C"), ("C", "D")])
    prep.G = G

    sources, sinks = prep.prepare_bioflow_topology(ligands=["A"], target_genes=["D"])

    # lock_targets=3: both original source_nodes (B) and sink_nodes (D) should be locked
    assert "B" in sources
    assert "D" in sinks


# --- Percolation stat/norm tests (bulk expression support) ---


def _prep_with_anndata(mock_anndata):
    prep = BioNetworkPreparer()
    prep.expression = mock_anndata
    prep.annotation = pd.Series(["type"] * 5, index=list("ABCDE"))
    return prep


@pytest.mark.parametrize(
    "stat,expected",
    [
        ("mean", {"A": 1.0, "B": 0.0, "C": 0.5, "D": 0.25, "E": 0.75}),
        ("median", {"A": 1.0, "B": 0.0, "C": 0.5, "D": 0.0, "E": 1.0}),
        ("max", {"A": 1.0, "B": 0.0, "C": 1.0, "D": 1.0, "E": 1.0}),
        ("p90", {"A": 1.0, "B": 0.0, "C": 1.0, "D": 0.7, "E": 1.0}),
    ],
)
def test_percolation_stats(mock_anndata, stat, expected):
    prep = _prep_with_anndata(mock_anndata)
    percolation = prep.compute_percolation(stat=stat)
    for gene, value in expected.items():
        assert percolation[gene] == pytest.approx(value)


@pytest.mark.parametrize(
    "norm,expected",
    [
        # Row sums are [2, 3, 4, 1]; CPM per gene then divided by max (gene A)
        ("cpm", {"A": 1.0, "B": 0.0, "C": 0.28, "D": 0.12, "E": 0.52}),
        # Data is binary, so log1p(mean) mirrors the raw mean profile
        ("log1p", {"A": 1.0, "B": 0.0, "C": 0.5, "D": 0.25, "E": 0.75}),
        # Constant genes (A, B) collapse to 0 under per-gene min-max
        ("minmax", {"A": 0.0, "B": 0.0, "C": 2 / 3, "D": 1 / 3, "E": 1.0}),
    ],
)
def test_percolation_mean_with_norm(mock_anndata, norm, expected):
    prep = _prep_with_anndata(mock_anndata)
    percolation = prep.compute_percolation(stat="mean", norm=norm)
    for gene, value in expected.items():
        assert percolation[gene] == pytest.approx(value)
    assert all(0.0 <= v <= 1.0 for v in percolation.values())


def test_nonzero_ratio_ignores_norm(mock_anndata):
    prep = _prep_with_anndata(mock_anndata)
    baseline = prep.compute_percolation(stat="nonzero_ratio", norm="none")
    for norm in ("cpm", "log1p", "minmax"):
        assert prep.compute_percolation(stat="nonzero_ratio", norm=norm) == baseline


def test_percolation_zero_matrix_gives_zeros():
    prep = BioNetworkPreparer()
    prep.expression = pd.DataFrame(
        [[0, 0], [0, 0]], columns=["A", "B"], index=["S1", "S2"]
    )
    prep.annotation = pd.Series(["type"] * 2, index=["A", "B"])
    percolation = prep.compute_percolation()
    assert percolation == {"A": 0.0, "B": 0.0}


def test_percolation_without_expression_is_falsy(mock_annotation):
    prep = BioNetworkPreparer()
    prep.annotation = mock_annotation
    assert prep.compute_percolation() is False


def test_percolation_zero_overlap_prints_warning(mock_annotation, capsys):
    prep = BioNetworkPreparer()
    prep.annotation = mock_annotation
    prep.expression = pd.DataFrame([[1, 2]], columns=["X", "Y"])
    percolation = prep.compute_percolation()
    assert percolation == {g: 0.0 for g in mock_annotation.index}
    assert "shares no genes" in capsys.readouterr().out


def test_compute_percolation_rejects_invalid_args(mock_anndata):
    prep = _prep_with_anndata(mock_anndata)
    with pytest.raises(ValueError, match="expression_stat"):
        prep.compute_percolation(stat="mode")
    with pytest.raises(ValueError, match="expression_norm"):
        prep.compute_percolation(norm="quantile")


# --- normalize_expression / transpose_expression units ---


def test_normalize_expression_cpm():
    df = pd.DataFrame([[1, 1], [0, 0]], columns=["A", "B"], index=["S1", "S2"])
    out = normalize_expression(df, "cpm")
    assert out.loc["S1"].tolist() == pytest.approx([500_000.0, 500_000.0])
    assert out.loc["S2"].tolist() == pytest.approx([0.0, 0.0])


def test_normalize_expression_log1p():
    df = pd.DataFrame([[0.0, 1.0]], columns=["A", "B"])
    out = normalize_expression(df, "log1p")
    assert out.iloc[0, 0] == 0.0
    assert out.iloc[0, 1] == pytest.approx(np.log(2))


def test_normalize_expression_minmax():
    df = pd.DataFrame([[0.0, 5.0, 3.0], [10.0, 5.0, 7.0]], columns=["A", "B", "C"])
    out = normalize_expression(df, "minmax")
    assert out["A"].tolist() == pytest.approx([0.0, 1.0])
    # Constant gene -> all zeros (no division by zero)
    assert out["B"].tolist() == pytest.approx([0.0, 0.0])
    assert out["C"].tolist() == pytest.approx([0.0, 1.0])


def test_normalize_expression_none_passthrough():
    df = pd.DataFrame([[1.0]])
    assert normalize_expression(df, "none") is df


def test_normalize_expression_rejects_unknown_method():
    with pytest.raises(ValueError, match="expression_norm"):
        normalize_expression(pd.DataFrame([[1.0]]), "quantile")


def test_transpose_expression_with_gene_column():
    df = pd.DataFrame({"gene": ["A", "B"], "S1": [1, 0], "S2": [0, 2]})
    out = transpose_expression(df)
    assert list(out.index) == ["S1", "S2"]
    assert list(out.columns) == ["A", "B"]
    assert out.loc["S1", "A"] == 1


def test_transpose_expression_all_numeric_is_plain_transpose():
    df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
    out = transpose_expression(df)
    assert out.shape == (3, 2)
    assert out.iloc[0, 0] == 1
    assert out.iloc[2, 1] == 6


# --- load_data orientation tests (real files) ---


def _write_minimal_io(tmp_path):
    (tmp_path / "net.tsv").write_text(
        "start\tto\tweight\tlayer\nA\tB\t1.0\tactivation\n"
    )
    (tmp_path / "ann.tsv").write_text("gene\tnode_type\nA\tligand\nB\ttarget\n")


def _read_expr(tmp_path, expression_transpose):
    _write_minimal_io(tmp_path)
    prep = BioNetworkPreparer()
    prep.load_data(
        str(tmp_path / "net.tsv"),
        str(tmp_path / "ann.tsv"),
        str(tmp_path / "expr.csv"),
        expression_transpose=expression_transpose,
    )
    return prep.expression


def test_load_data_transpose_genes_as_rows(tmp_path):
    (tmp_path / "expr.csv").write_text("gene,S1,S2\nA,1,0\nB,0,2\n")
    df = _read_expr(tmp_path, expression_transpose=True)
    assert list(df.index) == ["S1", "S2"]
    assert list(df.columns) == ["A", "B"]
    assert df.loc["S1", "A"] == 1


def test_load_data_transpose_unnamed_first_column(tmp_path):
    (tmp_path / "expr.csv").write_text(",S1,S2\nA,1,0\nB,0,2\n")
    df = _read_expr(tmp_path, expression_transpose=True)
    assert list(df.index) == ["S1", "S2"]
    assert list(df.columns) == ["A", "B"]


def test_load_data_transpose_all_numeric_first_column(tmp_path):
    (tmp_path / "expr.csv").write_text("1,2,3\n4,5,6\n7,8,9\n")
    df = _read_expr(tmp_path, expression_transpose=True)
    assert df.shape == (3, 2)
    assert df.iloc[0, 0] == 4
    assert df.iloc[2, 1] == 9


def test_load_data_default_orientation_unchanged(tmp_path):
    (tmp_path / "expr.csv").write_text("gene,S1,S2\nA,1,0\nB,0,2\n")
    df = _read_expr(tmp_path, expression_transpose=False)
    assert list(df.columns) == ["gene", "S1", "S2"]
    assert len(df) == 2


def test_load_data_rejects_transpose_with_h5ad(tmp_path):
    _write_minimal_io(tmp_path)
    prep = BioNetworkPreparer()
    with pytest.raises(ValueError, match="only supported"):
        prep.load_data(
            str(tmp_path / "net.tsv"),
            str(tmp_path / "ann.tsv"),
            str(tmp_path / "expr.h5ad"),
            expression_transpose=True,
        )


def test_prepare_network_rejects_half_label_pair():
    prep = BioNetworkPreparer()
    with pytest.raises(ValueError, match="together"):
        prep.prepare_network(
            network_path="net",
            annotation_path="ann",
            expression_path="expr",
            ligands=["A"],
            targets=["B"],
            label="celltype",
        )


def test_prepare_network_rejects_invalid_stat_and_norm():
    prep = BioNetworkPreparer()
    with pytest.raises(ValueError, match="expression_stat"):
        prep.prepare_network(
            network_path="net",
            annotation_path="ann",
            expression_path="expr",
            ligands=["A"],
            targets=["B"],
            expression_stat="mode",
        )
    with pytest.raises(ValueError, match="expression_norm"):
        prep.prepare_network(
            network_path="net",
            annotation_path="ann",
            expression_path="expr",
            ligands=["A"],
            targets=["B"],
            expression_norm="quantile",
        )


def test_percolation_values_are_python_floats(mock_anndata):
    """AnnData X is float32; np.float32 is not JSON-serializable (cyjs saver)."""
    prep = _prep_with_anndata(mock_anndata)
    for stat, norm in [
        ("nonzero_ratio", "none"),
        ("mean", "cpm"),
        ("median", "log1p"),
        ("max", "minmax"),
    ]:
        percolation = prep.compute_percolation(stat=stat, norm=norm)
        # np.float32 is not a float subclass, so isinstance filters it out
        assert all(isinstance(v, float) for v in percolation.values())
