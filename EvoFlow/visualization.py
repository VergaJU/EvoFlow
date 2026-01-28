import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns
import pandas as pd
import numpy as np
import networkx as nx
import json

def plot_optimization_history(logs, filename_prefix):
    """
    Plots the average fitness with min-max range across generations.
    """
    if isinstance(logs, dict):
        logs_df = pd.DataFrame(logs).T
        logs_df['min'] = logs_df['min'].apply(lambda x: float(x[0]))
        logs_df['max'] = logs_df['max'].apply(lambda x: float(x[0]))
        logs_df = logs_df.reset_index(names='gen')
    else:
        logs_df = logs # Assume already dataframe

    plt.figure(figsize=(8, 5))
    sns.lineplot(x='gen', y='avg', data=logs_df, marker='o', label='Average', color='b')
    plt.fill_between(logs_df['gen'], logs_df['min'], logs_df['max'], color='b', alpha=0.1)

    plt.xlabel('Generation')
    plt.ylabel('Value')
    plt.title('Average with Min-Max Range Across Generations')
    plt.grid(True)
    plt.legend()
    
    fig_name = filename_prefix + "optimization_values.png"
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

def plot_weights(net_df, filename_prefix):
    """
    Plots weight rank and distribution.
    """
    weight_sequence = net_df["weight"].sort_values(ascending=False).values

    fig, (ax1, ax2)= plt.subplots(1,2, figsize=(16,9))
    
    # Rank Plot
    ax1.plot(weight_sequence, "b-", marker="o")
    ax1.set_title("Weight Rank Plot")
    quantile_90 = net_df["weight"].quantile(0.9)
    ax1.hlines(quantile_90, 0, len(weight_sequence), colors="red")
    ax1.text(len(weight_sequence)/2, quantile_90*1.1, "90th percentile", color="red")
    ax1.axhspan(0, quantile_90, facecolor='red', alpha=0.3)
    ax1.axhspan(quantile_90, net_df["weight"].max(), facecolor='green', alpha=0.3)
    ax1.set_ylabel("Weight")
    ax1.set_xlabel("Rank")

    # Histogram
    hist, bin_edges, _ = ax2.hist(weight_sequence, bins=40)
    ax2.set_title("Weight histogram")
    ax2.vlines(quantile_90, 0, np.max(hist), colors="red")
    ax2.text(quantile_90*1.1, np.max(hist)/2, "90th percentile", color="red")
    ax2.axvspan(0, quantile_90, facecolor='red', alpha=0.3)
    ax2.axvspan(quantile_90, net_df["weight"].max(), facecolor='green', alpha=0.3)
    ax2.set_xlabel("Weight")
    ax2.set_ylabel("# of edges")

    fig.tight_layout()
    fig_name = filename_prefix + "weights.png"
    plt.savefig(fig_name)
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
