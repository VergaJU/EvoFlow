"""Plotting and network-export helpers.

Every function writes its figure/network to the **current working directory**
by concatenating ``filename_prefix`` with a fixed suffix (for example
``<prefix>optimization_values.png`` or ``<prefix>network.graphml``). The CLI
passes a prefix derived from the expression file name.
"""

import json

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_hex
from matplotlib.ticker import PercentFormatter


def plot_optimization_history(logs, filename_prefix):
    """Plot the evolution of metrics during a GA run.

    The function accepts either a :class:`pandas.DataFrame` or a dictionary
    keyed by generation. Each entry should itself be a mapping from metric name
    to value; tuple-valued entries will be expanded into separate columns.
    Typical keys include ``Flow``, ``Node``, ``Redundancy`` and ``hv``, but any
    numeric metric present in the logs will be plotted.

    A grid of subplots is generated automatically based on the number of
    metrics found. The x-axis for every subplot is the generation number
    (``gen``); the y-axis is the metric value. The resulting figure is saved as
    ``<filename_prefix>optimization_values.png``.

    Args:
        logs (dict or pandas.DataFrame): Optimization history. If a dict, keys
            should be generation numbers and values metric dictionaries. If a
            DataFrame, a ``gen`` column should exist or will be inferred from
            the index.
        filename_prefix (str): Prefix used when saving the figure.

    Returns:
        matplotlib.figure.Figure: The created figure (also saved to disk).

    Raises:
        ValueError: If the logs contain no plottable metrics.
    """

    # --- normalize input to DataFrame ---
    if isinstance(logs, dict):
        logs_df = pd.DataFrame(logs).T

        logs_df = logs_df.reset_index(names="gen")
    else:
        logs_df = logs.copy()
        if "gen" not in logs_df.columns:
            logs_df = logs_df.reset_index(names="gen")
    # summary_logs (returned by GAOptimizer.run) exclude visited_nodes;
    # expanded logs include it. Drop it if present.
    logs_df = logs_df.drop("visited_nodes", axis=1, errors="ignore")
    # choose the order of known metrics first so plots are predictable
    known_order = [
        "Flow",
        "Node",
        "Redundancy",
        "hv",
        "hd",
        "cd",
        "cxpb",
        "mtpb",
        "indpb",
    ]
    all_metrics = [c for c in logs_df.columns if c != "gen"]
    value_cols = [c for c in known_order if c in all_metrics]
    # append any additional metrics that were not in the known list
    value_cols += [c for c in all_metrics if c not in value_cols]

    n_metrics = len(value_cols)
    if n_metrics == 0:
        raise ValueError("No metrics found in logs to plot")

    # determine subplot grid dimensions (max 3 columns for readability)
    import math

    ncols = min(3, n_metrics)
    nrows = math.ceil(n_metrics / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    # make axes iterable
    if isinstance(axes, np.ndarray):
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]

    # helper to convert metric name to a human friendly title
    title_map = {
        "hv": "Hypervolume",
        "cd": "Crowding Distance",
        "hd": "Hamming Distance",
        "Flow": "Flow",
        "Node": "Nodes",
        "Redundancy": "Redundancy",
    }

    for idx, col in enumerate(value_cols):
        ax = axes_flat[idx]
        sns.lineplot(x="gen", y=col, data=logs_df, ax=ax)
        title = title_map.get(col, col.capitalize())
        ax.set_title(title)
        ax.set_xlabel("Generation")
        ax.set_ylabel(title)
        ax.grid(True)

    # turn off any unused axes
    for ax in axes_flat[n_metrics:]:
        ax.axis("off")

    fig_name = filename_prefix + "optimization_values.png"
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close(fig)
    return fig


def plot_node_proportions(annotation, filename_prefix):
    """Plot the proportion of each node type.

    Args:
        annotation (pandas.Series): Gene -> annotation table produced by
            :class:`EvoFlow.biology.BioNetworkPreparer`.
        filename_prefix (str): Prefix used when saving the figure; the output is
            ``<filename_prefix>nodes_proportions.png``.
    """
    annotation_prop = annotation.value_counts() / (annotation.value_counts().sum())

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(list(annotation_prop.index), annotation_prop)
    for index, data in enumerate(annotation_prop):
        ax.text(
            x=index,
            y=data,
            s=f"{int(data * annotation.value_counts().sum())}",
            fontdict=dict(fontsize=12),
        )

    ax.set_title("Proportions node types")
    ax.yaxis.set_major_formatter(PercentFormatter(1))

    fig_name = filename_prefix + "nodes_proportions.png"
    plt.savefig(fig_name)
    plt.close()


def plot_node_attribute(
    G,
    filename_prefix,
    attribute: str = "percolation",
    show: bool = True,
    save: bool = True,
):
    """Plot the rank distribution and histogram of a node attribute.

    Args:
        G (networkx.DiGraph): Graph whose nodes carry ``attribute``.
        filename_prefix (str): Prefix for the saved figure; the output is
            ``<filename_prefix><attribute>.png``.
        attribute (str): Node attribute to plot (e.g. ``"percolation"``).
            Defaults to ``"percolation"``.
        show (bool): Whether to display the figure interactively. Defaults to
            ``True``.
        save (bool): Whether to write the figure to disk. Defaults to ``True``.

    Raises:
        ValueError: If ``attribute`` is missing from a node, is non-numeric, or
            has no finite values.
    """
    # Extract attribute values from nodes
    attribute_values = []
    for node, attrs in G.nodes(data=True):
        if attribute not in attrs:
            raise ValueError(
                f"Attribute '{attribute}' not found in node '{node}'. "
                f"Available attributes: {set().union(*[set(attrs.keys()) for _, attrs in G.nodes(data=True)])}"
            )
        value = attrs[attribute]
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"Attribute '{attribute}' in node '{node}' has non-numeric value: {value}. "
                f"Only numeric attributes are supported."
            )
        if not np.isfinite(value):
            continue
        attribute_values.append(value)

    if not attribute_values:
        raise ValueError(f"Attribute '{attribute}' has no finite values to plot.")

    weight_sequence = sorted(attribute_values, reverse=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))

    # Rank Plot
    ax1.plot(weight_sequence, "b-", marker="o")
    ax1.set_title(f"{attribute.capitalize()} Rank Plot")
    quantile_90 = np.quantile(weight_sequence, 0.9)
    ax1.hlines(quantile_90, 0, len(weight_sequence), colors="red")
    ax1.text(
        len(weight_sequence) / 2, quantile_90 * 1.1, "90th percentile", color="red"
    )
    ax1.axhspan(0, quantile_90, facecolor="red", alpha=0.3)
    ax1.axhspan(quantile_90, max(weight_sequence), facecolor="green", alpha=0.3)
    ax1.set_ylabel(attribute.capitalize())
    ax1.set_xlabel("Rank")

    # Histogram
    hist, bin_edges, _ = ax2.hist(weight_sequence, bins=40)
    ax2.set_title(f"{attribute.capitalize()} Histogram")
    ax2.vlines(quantile_90, 0, np.max(hist), colors="red")
    ax2.text(quantile_90 * 1.1, np.max(hist) / 2, "90th percentile", color="red")
    ax2.axvspan(0, quantile_90, facecolor="red", alpha=0.3)
    ax2.axvspan(quantile_90, max(weight_sequence), facecolor="green", alpha=0.3)
    ax2.set_xlabel(attribute.capitalize())
    ax2.set_ylabel("# of nodes")

    fig.tight_layout()
    if save:
        fig_name = filename_prefix + f"{attribute}.png"
        plt.savefig(fig_name)

    if show:
        plt.show()
    else:
        plt.close()


def save_network(G, filename_prefix):
    """Save the network as Cytoscape JSON and GraphML.

    Args:
        G (networkx.DiGraph): Graph to export.
        filename_prefix (str): Prefix for the output files;
            ``<prefix>network.cyjs`` and ``<prefix>network.graphml`` are written
            to the current working directory.
    """
    # Cytoscape JSON
    cs = nx.cytoscape_data(G)
    graph_name = filename_prefix + "network.cyjs"
    with open(graph_name, "w") as outfile:
        json.dump(cs, outfile)

    # GraphML
    graph_name = filename_prefix + "network.graphml"
    nx.write_graphml(G, graph_name)


def plot_graph(
    G,
    filename_prefix=None,
    layout="kamada_kawai",
    figsize=(15, 12),
    *,
    layout_kwargs=None,
    node_size=300,
    font_size=8,
    with_labels=True,
    edge_alpha=0.6,
):
    """Plot the network with nodes colored by type.

    Nodes are grouped by their ``node_type`` attribute, and each group gets one
    color from the ``tab10`` palette plus one entry in the legend. The figure is
    always returned so it can be shown in a notebook; a PNG copy is written only
    when ``filename_prefix`` is given.

    Args:
        G (networkx.DiGraph): Graph to draw. Node colors come from the
            ``node_type`` node attribute (``"unknown"`` when absent).
        filename_prefix (str, optional): Prefix for the saved figure; output is
            ``<prefix>network_plot_<layout>.png``. If ``None`` the figure is not
            written to disk.
        layout (str): Layout algorithm, one of ``"kamada_kawai"`` (default),
            ``"spring"``, ``"circular"``, ``"shell"``, ``"spectral"`` or
            ``"spiral"``.
        figsize (tuple): Figure size ``(width, height)`` in inches. Defaults to
            ``(15, 12)``.
        layout_kwargs (dict, optional): Keyword arguments forwarded to the
            selected NetworkX layout function. Use it to make stochastic
            layouts reproducible, for example ``{"k": 0.8, "iterations": 100,
            "seed": 42}`` with ``layout="spring"``.
        node_size (float or array-like): Node size passed to
            ``networkx.draw_networkx_nodes``. Defaults to ``300``. Use smaller
            values (for example ``100``) for dense networks.
        font_size (int): Label font size. Defaults to ``8``.
        with_labels (bool): Whether to draw node labels. Defaults to ``True``;
            disable it for large networks where labels overlap.
        edge_alpha (float): Edge opacity from 0 to 1. Defaults to ``0.6``.

    Returns:
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The figure and
        axes objects.

    Raises:
        ValueError: If ``layout`` is not one of the supported algorithms.

    Examples:
        >>> import networkx as nx
        >>> from EvoFlow.visualization import plot_graph
        >>> G = nx.DiGraph()
        >>> G.add_node("A", node_type="gene")
        >>> G.add_node("B", node_type="protein")
        >>> G.add_edge("A", "B")
        >>> fig, ax = plot_graph(G, layout="spring", figsize=(4, 4))
    """
    # Define layout algorithms
    layout_functions = {
        "kamada_kawai": nx.kamada_kawai_layout,
        "spring": nx.spring_layout,
        "circular": nx.circular_layout,
        "shell": nx.shell_layout,
        "spectral": nx.spectral_layout,
        "spiral": nx.spiral_layout,
    }

    if layout not in layout_functions:
        raise ValueError(
            f"Layout '{layout}' not supported. Choose from: {list(layout_functions.keys())}"
        )

    # Compute layout
    pos = layout_functions[layout](G, **(layout_kwargs or {}))

    # Get unique node types and assign colors
    node_types = {}
    for node, attrs in G.nodes(data=True):
        node_type = attrs.get("node_type", "unknown")
        if node_type not in node_types:
            node_types[node_type] = []
        node_types[node_type].append(node)

    # Define a color palette for node types. Hex strings are passed instead of
    # RGB(A) tuples: networkx otherwise interprets a tuple as a value-mapping
    # and emits a UserWarning.
    colors_palette = plt.get_cmap("tab10")
    color_map = {}
    for idx, node_type in enumerate(sorted(node_types.keys())):
        color_map[node_type] = to_hex(colors_palette(idx % 10))

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Draw edges
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edge_color="gray",
        arrows=True,
        arrowsize=10,
        arrowstyle="->",
        width=0.5,
        alpha=edge_alpha,
        connectionstyle="arc3,rad=0.1",
    )

    # Draw nodes by type with different colors
    for node_type, nodes in node_types.items():
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodes,
            node_color=color_map[node_type],
            node_size=node_size,
            label=node_type,
            ax=ax,
            alpha=0.9,
        )

    # Draw labels
    if with_labels:
        nx.draw_networkx_labels(G, pos, font_size=font_size, font_weight="bold", ax=ax)

    # Formatting
    ax.set_title(f"Network Graph ({layout} layout)", fontsize=16, fontweight="bold")
    if node_types:
        ax.legend(scatterpoints=1, loc="upper left", fontsize=10)
    ax.axis("off")

    plt.tight_layout()

    # Save figure if filename_prefix is provided
    if filename_prefix:
        fig_name = filename_prefix + f"network_plot_{layout}.png"
        plt.savefig(fig_name, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_edge_attribute(
    G, filename_prefix, attribute: str = "weight", show: bool = True, save: bool = True
):
    """Plot the rank distribution and histogram of an edge attribute.

    Non-finite values (infinite capacities on super-node edges) are skipped so
    they do not dominate the histogram.

    Args:
        G (networkx.DiGraph): Graph whose edges carry ``attribute``.
        filename_prefix (str): Prefix for the saved figure; the output is
            ``<filename_prefix>edge_<attribute>.png``.
        attribute (str): Edge attribute to plot (e.g. ``"weight"`` or
            ``"capacity"``). Defaults to ``"weight"``.
        show (bool): Whether to display the figure interactively. Defaults to
            ``True``.
        save (bool): Whether to write the figure to disk. Defaults to ``True``.

    Raises:
        ValueError: If ``attribute`` is missing from an edge, is non-numeric, or
            has no finite values.
    """
    # Extract attribute values from edges
    attribute_values = []
    for u, v, attrs in G.edges(data=True):
        if attribute not in attrs:
            raise ValueError(
                f"Attribute '{attribute}' not found in edge ({u}, {v}). "
                f"Available attributes: {set().union(*[set(attrs.keys()) for _, _, attrs in G.edges(data=True)])}"
            )
        value = attrs[attribute]
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"Attribute '{attribute}' in edge ({u}, {v}) has non-numeric value: {value}. "
                f"Only numeric attributes are supported."
            )
        # Dummy/sentinel edges carry inf capacities by design; exclude them
        # from distributions (np.histogram rejects non-finite ranges).
        if not np.isfinite(value):
            continue
        attribute_values.append(value)

    if not attribute_values:
        raise ValueError(f"Attribute '{attribute}' has no finite values to plot.")

    weight_sequence = sorted(attribute_values, reverse=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))

    # Rank Plot
    ax1.plot(weight_sequence, "b-", marker="o")
    ax1.set_title(f"Edge {attribute.capitalize()} Rank Plot")
    quantile_90 = np.quantile(weight_sequence, 0.9)
    ax1.hlines(quantile_90, 0, len(weight_sequence), colors="red")
    ax1.text(
        len(weight_sequence) / 2, quantile_90 * 1.1, "90th percentile", color="red"
    )
    ax1.axhspan(0, quantile_90, facecolor="red", alpha=0.3)
    ax1.axhspan(quantile_90, max(weight_sequence), facecolor="green", alpha=0.3)
    ax1.set_ylabel(attribute.capitalize())
    ax1.set_xlabel("Rank")

    # Histogram
    hist, bin_edges, _ = ax2.hist(weight_sequence, bins=40)
    ax2.set_title(f"Edge {attribute.capitalize()} Histogram")
    ax2.vlines(quantile_90, 0, np.max(hist), colors="red")
    ax2.text(quantile_90 * 1.1, np.max(hist) / 2, "90th percentile", color="red")
    ax2.axvspan(0, quantile_90, facecolor="red", alpha=0.3)
    ax2.axvspan(quantile_90, max(weight_sequence), facecolor="green", alpha=0.3)
    ax2.set_xlabel(attribute.capitalize())
    ax2.set_ylabel("# of edges")

    fig.tight_layout()
    if save:
        fig_name = filename_prefix + f"edge_{attribute}.png"
        plt.savefig(fig_name)

    if show:
        plt.show()
    else:
        plt.close()
