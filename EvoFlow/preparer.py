"""Flow-topology preparation: add SuperSource, SuperSink and dummy nodes.

The dummy nodes decouple the biological source/sink genes from the artificial
super nodes so that the genetic algorithm can remove edges incident to a
receptor or target without disconnecting the whole network.
"""

import numpy as np


def prepare_flow_topology(G, source_nodes, sink_nodes, lock_targets: int = 0):
    """
    Prepare a graph for flow analysis by adding SuperSource, SuperSink, and dummy nodes.

    The graph ``G`` is modified in place: a ``SuperSource`` is connected to each
    source node through a ``dummy_<node>`` intermediary, and each sink node is
    connected to ``SuperSink`` through its own ``dummy_<node>``. Dummy edges use
    the mean capacity of the existing edges; super-node edges are infinite.

    Args:
        G (networkx.DiGraph): The graph to modify in place.
        source_nodes (list): Node names to treat as sources.
        sink_nodes (list): Node names to treat as sinks. Nodes missing from
            ``G`` are silently skipped.
        lock_targets (int): Bitmask controlling which original nodes are locked
            into the returned source/sink sets (and therefore protected from
            removal by the GA).
            0 (default): No locking.
            1: ``source_nodes`` are added to ``final_sources``.
            2: ``sink_nodes`` are added to ``final_sinks``.
            3: Both are locked.

    Returns:
        tuple[list, list]: ``(final_sources, final_sinks)`` where
        ``final_sources`` includes ``"SuperSource"`` plus the dummy sources and
        ``final_sinks`` includes the dummy sinks plus ``"SuperSink"``.

    Raises:
        ValueError: If ``lock_targets`` is not one of ``0, 1, 2, 3``.

    Examples:
        >>> import networkx as nx
        >>> from EvoFlow.preparer import prepare_flow_topology
        >>> G = nx.DiGraph()
        >>> G.add_edge("L", "R", capacity=1.0)
        >>> sources, sinks = prepare_flow_topology(G, ["R"], ["T"])
        >>> "SuperSource" in sources
        True
    """
    if lock_targets not in (0, 1, 2, 3):
        raise ValueError(f"lock_targets must be 0, 1, 2, or 3, got {lock_targets}")

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
    if lock_targets == 1 or lock_targets == 3:
        final_sources.extend(source_nodes)
    final_sinks = [n[0] for n in dummy_sinks] + ["SuperSink"]
    if lock_targets == 2 or lock_targets == 3:
        final_sinks.extend(sink_nodes)

    return final_sources, final_sinks
