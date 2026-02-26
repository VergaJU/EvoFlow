import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns
import pandas as pd
import numpy as np
import networkx as nx
import json

def plot_optimization_history(logs, filename_prefix):
    """
    Plots optimization history across generations in a 2x2 subplot grid.
    The function assumes the data (either a DataFrame or a dict) contains
    metrics with tuple values to expand into separate columns, plus an
    optional 'hv' (hypervolume) metric.

    Layout:
    - Top-left (titled **Flow**): first tuple element
    - Top-right (titled **Nodes**): second tuple element
    - Bottom-left (titled **Redundancy**): third tuple element
    - Bottom-right (titled **Hypervolume**): hv metric (if available)

    The x-axis for all panels is the generation number (`gen`).
    """
    # normalize input to a DataFrame with a 'gen' column
    if isinstance(logs, dict):
        # each key is a generation, value is a dict of metrics
        logs_df = pd.DataFrame(logs).T

        # expand tuple values into separate columns
        for col in logs_df.columns:
            sample = logs_df[col].iloc[0]
            if isinstance(sample, (list, tuple)) and len(sample) > 1:
                # expand into multiple columns
                expanded = pd.DataFrame(logs_df[col].tolist(),
                                        index=logs_df.index)
                for i, newcol in enumerate(expanded.columns):
                    logs_df[f"{col}_{i}"] = expanded[newcol]
                logs_df = logs_df.drop(columns=[col])
        logs_df = logs_df.reset_index(names='gen')
    else:
        logs_df = logs.copy()
        if 'gen' not in logs_df.columns:
            logs_df = logs_df.reset_index(names='gen')

    # determine data columns to plot
    value_cols = [c for c in logs_df.columns if c != 'gen']
    titles = ['Hypervolume', 'Flow', 'Nodes', 'Redundancy']
    
    # create 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.flatten()
    
    for ax, title, col in zip(axes_flat, titles, value_cols[:4]):
        if col in logs_df.columns:
            sns.lineplot(x='gen', y=col, data=logs_df, marker='o', ax=ax)
            ax.set_title(title)
            ax.set_xlabel('Generation')
            ax.set_ylabel(title)
            ax.grid(True)
        else:
            ax.axis('off')

    fig_name = filename_prefix + "optimization_values.png"
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close()

def plot_node_proportions(annotation, filename_prefix):
    """
    Plots the proportion of node types.
    """
    annotation_prop = (annotation.value_counts()/(annotation.value_counts().sum()))

    fig, ax = plt.subplots(figsize=(10,5))
    ax.bar(list(annotation_prop.index), annotation_prop)
    for index, data in enumerate(annotation_prop):
        ax.text(x=index , y=data , s=f"{int(data*annotation.value_counts().sum())}" , fontdict=dict(fontsize=12))
        
    ax.set_title("Proportions node types")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    
    fig_name = filename_prefix + "nodes_proportions.png"
    plt.savefig(fig_name)
    plt.close()

def plot_node_attribute(G, 
                 filename_prefix, 
                 attribute: str = 'percolation', 
                 show: bool=True, 
                 save: bool = True):
    """
    Plots node attribute rank and distribution.
    
    Parameters:
    -----------
    G : nx.DiGraph
        The network graph containing nodes with attributes.
    filename_prefix : str
        Prefix for saving the figure.
    attribute : str, default 'percolation'
        The node attribute to plot (e.g., 'percolation', 'capacity').
    
    Raises:
    -------
    ValueError
        If the specified attribute does not exist or contains non-numeric values.
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
            attribute_values.append(float(value))
        except (TypeError, ValueError):
            raise ValueError(
                f"Attribute '{attribute}' in node '{node}' has non-numeric value: {value}. "
                f"Only numeric attributes are supported."
            )
    
    weight_sequence = sorted(attribute_values, reverse=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
    
    # Rank Plot
    ax1.plot(weight_sequence, "b-", marker="o")
    ax1.set_title(f"{attribute.capitalize()} Rank Plot")
    quantile_90 = np.quantile(weight_sequence, 0.9)
    ax1.hlines(quantile_90, 0, len(weight_sequence), colors="red")
    ax1.text(len(weight_sequence)/2, quantile_90*1.1, "90th percentile", color="red")
    ax1.axhspan(0, quantile_90, facecolor='red', alpha=0.3)
    ax1.axhspan(quantile_90, max(weight_sequence), facecolor='green', alpha=0.3)
    ax1.set_ylabel(attribute.capitalize())
    ax1.set_xlabel("Rank")

    # Histogram
    hist, bin_edges, _ = ax2.hist(weight_sequence, bins=40)
    ax2.set_title(f"{attribute.capitalize()} Histogram")
    ax2.vlines(quantile_90, 0, np.max(hist), colors="red")
    ax2.text(quantile_90*1.1, np.max(hist)/2, "90th percentile", color="red")
    ax2.axvspan(0, quantile_90, facecolor='red', alpha=0.3)
    ax2.axvspan(quantile_90, max(weight_sequence), facecolor='green', alpha=0.3)
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
    """
    Saves the network in Cytoscape JSON and GraphML formats.
    """
    # Cytoscape JSON
    cs = nx.cytoscape_data(G)
    graph_name = filename_prefix + "network.cyjs"
    with open(graph_name, "w") as outfile:
        json.dump(cs, outfile)

    # GraphML
    graph_name = filename_prefix + "network.graphml"
    nx.write_graphml(G, graph_name)

def plot_graph(G, filename_prefix=None, layout="kamada_kawai", figsize=(15, 12)):
    """
    Plots the network graph with nodes colored by their type.
    
    Parameters:
    -----------
    G : nx.DiGraph
        The network graph to plot.
    filename_prefix : str, optional
        Prefix for saving the figure. If None, plot is not saved.
    layout : str, default "kamada_kawai"
        Layout algorithm to use. Options include:
        - "kamada_kawai": Kamada-Kawai layout (default)
        - "spring": Spring layout (Fruchterman-Reingold)
        - "circular": Circular layout
        - "shell": Shell layout
        - "spectral": Spectral layout
        - "spiral": Spiral layout
    figsize : tuple, default (15, 12)
        Figure size (width, height) in inches.
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.
    ax : matplotlib.axes.Axes
        The matplotlib axes object.
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
    pos = layout_functions[layout](G)
    
    # Get unique node types and assign colors
    node_types = {}
    for node, attrs in G.nodes(data=True):
        node_type = attrs.get("node_type", "unknown")
        if node_type not in node_types:
            node_types[node_type] = []
        node_types[node_type].append(node)
    
    # Define a color palette for node types
    colors_palette = plt.cm.get_cmap("tab10")
    color_map = {}
    for idx, node_type in enumerate(sorted(node_types.keys())):
        color_map[node_type] = colors_palette(idx % 10)
    
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
        alpha=0.6,
        connectionstyle="arc3,rad=0.1"
    )
    
    # Draw nodes by type with different colors
    for node_type, nodes in node_types.items():
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodes,
            node_color=color_map[node_type],
            node_size=300,
            label=node_type,
            ax=ax,
            alpha=0.9,
        )
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold", ax=ax)
    
    # Formatting
    ax.set_title(f"Network Graph ({layout} layout)", fontsize=16, fontweight="bold")
    ax.legend(scatterpoints=1, loc="upper left", fontsize=10)
    ax.axis("off")
    
    plt.tight_layout()
    
    # Save figure if filename_prefix is provided
    if filename_prefix:
        fig_name = filename_prefix + f"network_plot_{layout}.png"
        plt.savefig(fig_name, dpi=300, bbox_inches="tight")
    
    return fig, ax

def plot_edge_attribute(G, 
                      filename_prefix, 
                      attribute: str = 'weight', 
                      show: bool = True, 
                      save: bool = True):
    """
    Plots edge attribute rank and distribution.
    
    Parameters:
    -----------
    G : nx.DiGraph
        The network graph containing edges with attributes.
    filename_prefix : str
        Prefix for saving the figure.
    attribute : str, default 'weight'
        The edge attribute to plot (e.g., 'weight', 'capacity').
    show : bool, default True
        Whether to display the plot.
    save : bool, default True
        Whether to save the plot to a file.
    
    Raises:
    -------
    ValueError
        If the specified attribute does not exist or contains non-numeric values.
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
            attribute_values.append(float(value))
        except (TypeError, ValueError):
            raise ValueError(
                f"Attribute '{attribute}' in edge ({u}, {v}) has non-numeric value: {value}. "
                f"Only numeric attributes are supported."
            )
    
    weight_sequence = sorted(attribute_values, reverse=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
    
    # Rank Plot
    ax1.plot(weight_sequence, "b-", marker="o")
    ax1.set_title(f"Edge {attribute.capitalize()} Rank Plot")
    quantile_90 = np.quantile(weight_sequence, 0.9)
    ax1.hlines(quantile_90, 0, len(weight_sequence), colors="red")
    ax1.text(len(weight_sequence)/2, quantile_90*1.1, "90th percentile", color="red")
    ax1.axhspan(0, quantile_90, facecolor='red', alpha=0.3)
    ax1.axhspan(quantile_90, max(weight_sequence), facecolor='green', alpha=0.3)
    ax1.set_ylabel(attribute.capitalize())
    ax1.set_xlabel("Rank")

    # Histogram
    hist, bin_edges, _ = ax2.hist(weight_sequence, bins=40)
    ax2.set_title(f"Edge {attribute.capitalize()} Histogram")
    ax2.vlines(quantile_90, 0, np.max(hist), colors="red")
    ax2.text(quantile_90*1.1, np.max(hist)/2, "90th percentile", color="red")
    ax2.axvspan(0, quantile_90, facecolor='red', alpha=0.3)
    ax2.axvspan(quantile_90, max(weight_sequence), facecolor='green', alpha=0.3)
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


