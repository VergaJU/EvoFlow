import os
import numpy as np
import pandas as pd

from EvoFlow.visualization import plot_optimization_history


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
            "visited_nodes": ['a','b','c']
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
            "visited_nodes": ['d','e','f']
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
