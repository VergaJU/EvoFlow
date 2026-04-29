import numpy as np


def prepare_flow_topology(G, source_nodes, sink_nodes):
    """
    General function to prepare a graph for flow analysis by adding SuperSource, SuperSink, and dummy nodes.

    Args:
        G (nx.DiGraph): The graph to modify.
        source_nodes (list): List of node names to treat as sources (will connect to SuperSource via dummies).
        sink_nodes (list): List of node names to treat as sinks (will connect to SuperSink via dummies).

    Returns:
        tuple: (final_sources, final_sinks) where final_sources includes SuperSource and dummy sources,
               and final_sinks includes dummy sinks and SuperSink.
    """
    # Compute mean capacity from existing edges
    capacities = [d.get("capacity", 0) for u, v, d in G.edges(data=True)]
    mean_capacity = np.mean(capacities) if capacities else 1.0

    # Setup SuperSink and dummy sinks
    supersink = [("SuperSink", {"node_type": "Sink", "percolation": 0})]
    dummy_sinks = []
    dummy_sink_edges = []
    sink_edges = []

    for node in sink_nodes:
        if node in G:
            dummy_name = f"dummy_{node}"
            dummy_sinks.append((dummy_name, {"node_type": "dummy", "percolation": 1}))
            dummy_sink_edges.append(
                (
                    node,
                    dummy_name,
                    {"weight": 1, "edge_type": "sink", "capacity": mean_capacity},
                )
            )
            sink_edges.append(
                (
                    dummy_name,
                    "SuperSink",
                    {"weight": 1, "edge_type": "sink", "capacity": float("inf")},
                )
            )

    G.add_nodes_from(supersink)
    G.add_nodes_from(dummy_sinks)
    G.add_edges_from(dummy_sink_edges)
    G.add_edges_from(sink_edges)

    # Setup SuperSource and dummy sources
    supersource = [("SuperSource", {"node_type": "Source", "percolation": 0})]
    dummy_source_nodes = []
    dummy_source_edges = []
    source_edges = []

    for node in source_nodes:
        dummy_name = f"dummy_{node}"
        dummy_source_nodes.append(
            (dummy_name, {"node_type": "dummy", "percolation": 1})
        )
        dummy_source_edges.append(
            (
                dummy_name,
                node,
                {"weight": 1, "edge_type": "source", "capacity": mean_capacity},
            )
        )
        source_edges.append(
            (
                "SuperSource",
                dummy_name,
                {"weight": 1, "edge_type": "source", "capacity": float("inf")},
            )
        )

    G.add_nodes_from(supersource)
    G.add_nodes_from(dummy_source_nodes)
    G.add_edges_from(dummy_source_edges)
    G.add_edges_from(source_edges)

    # Define final sources and sinks
    final_sources = ["SuperSource"] + [n[0] for n in dummy_source_nodes]
    final_sinks = [n[0] for n in dummy_sinks] + ["SuperSink"]

    return final_sources, final_sinks
