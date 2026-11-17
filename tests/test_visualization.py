import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from EvoFlow.visualization import plot_graph, plot_optimization_history


def _make_sample_logs():
    """Return a small example logs dict with the new structure described by user."""
    # two generations with all the requested metrics
    return {
        0: {
            "Flow": np.float64(1.0),
            "Node": 2.0,
            "Redundancy": 3.0,
            "hv": np.float64(0.1),
            "hd": np.float64(0.01),
            "cd": np.float64(0.02),
            "cxpb": 0.5,
            "mtpb": 0.6,
            "indpb": 0.7,
            "visited_nodes": ["a", "b", "c"],
        },
        1: {
            "Flow": np.float64(1.1),
            "Node": 2.1,
            "Redundancy": 3.1,
            "hv": np.float64(0.15),
            "hd": np.float64(0.015),
            "cd": np.float64(0.025),
            "cxpb": 0.55,
            "mtpb": 0.65,
            "indpb": 0.75,
            "visited_nodes": ["d", "e", "f"],
        },
    }


def test_plot_history_dict(tmp_path):
    """Calling with a dict should create an image file without error."""
    logs = _make_sample_logs()
    prefix = os.path.join(str(tmp_path), "out_")
    fig = plot_optimization_history(logs, prefix)
    # returned object should be a matplotlib figure
    assert fig is not None
    expected_file = prefix + "optimization_values.png"
    assert os.path.isfile(expected_file), "output file was not created"
    # file should be non-empty
    assert os.path.getsize(expected_file) > 0


def test_plot_history_dataframe(tmp_path):
    """The function should also accept a DataFrame with a gen column."""
    logs = _make_sample_logs()
    df = pd.DataFrame(logs).T.reset_index(names="gen")
    prefix = os.path.join(str(tmp_path), "out2_")
    fig = plot_optimization_history(df, prefix)
    assert fig is not None
    expected_file = prefix + "optimization_values.png"
    assert os.path.isfile(expected_file)


def test_plot_graph_accepts_layout_and_readability_options(mocker):
    """Layout and drawing options should be passed to the corresponding helpers."""
    graph = nx.DiGraph()
    graph.add_node("a", node_type="gene")
    graph.add_node("b", node_type="protein")
    graph.add_edge("a", "b")
    draw_edges = mocker.patch(
        "EvoFlow.visualization.nx.draw_networkx_edges",
        wraps=nx.draw_networkx_edges,
    )
    layout = mocker.patch(
        "EvoFlow.visualization.nx.spring_layout",
        return_value={"a": (0, 0), "b": (1, 1)},
    )

    fig, ax = plot_graph(
        graph,
        layout="spring",
        layout_kwargs={"k": 0.8, "iterations": 100, "seed": 42},
        node_size=120,
        font_size=6,
        with_labels=False,
        edge_alpha=0.25,
    )

    layout.assert_called_once_with(graph, k=0.8, iterations=100, seed=42)
    assert draw_edges.call_args.kwargs["alpha"] == 0.25
    assert not ax.texts
    assert all(collection.get_sizes()[0] == 120 for collection in ax.collections)
    plt.close(fig)
