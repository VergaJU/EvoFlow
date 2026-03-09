from EvoFlow.core import (
    calculate_flow_capacity,
    extract_active_subgraph,
    generate_individual,
    repair_individual,
    evaluate,
    GAOptimizer,
)
from deap import base, creator

# Ensure creator is setup if running tests in isolation
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)


def test_flow_calculation(simple_graph):
    # Path: S -> A(10) -> B(5) -> T(10)
    # Bottleneck is A->B with capacity 5.0
    flow = calculate_flow_capacity(simple_graph, "SuperSource", "SuperSink")
    assert flow == 5.0


def test_subgraph_extraction(branching_graph):
    # S->A->T, S->B->T. Both valid.
    # Add a dead end: S->C
    branching_graph.add_node("C")
    branching_graph.add_edge("SuperSource", "C")

    sub = extract_active_subgraph(branching_graph, "SuperSource", "SuperSink")

    assert "A" in sub.nodes
    assert "B" in sub.nodes
    assert "C" not in sub.nodes  # C is dead end, no path to sink


def test_generate_individual():
    removable_nodes = ["A", "B", "C", "D", "E"]
    ind = generate_individual(creator.Individual, removable_nodes, max_remove=3)

    # Check length
    assert len(ind) == 5
    # Check it's binary
    assert all(x in [0, 1] for x in ind)
    # Check number of zeros (removals) is between 1 and 3
    zeros = sum(1 for x in ind if x == 0)
    assert 1 <= zeros <= 3


def test_repair_individual():
    # Individual: [0, 0, 0, 0, 0] -> 5 removals
    # Max allowed: 3. Should flip 2 zeros to 1.
    ind = [0, 0, 0, 0, 0]
    # Note: repair_individual takes min_remove and max_remove (not min_removed/max_removed)
    repaired = repair_individual(ind, min_remove=1, max_remove=3)
    zeros = sum(1 for x in repaired if x == 0)
    assert zeros == 3

    # Individual: [1, 1, 1, 1, 1] -> 0 removals
    # Min allowed: 2. Should flip 2 ones to 0.
    ind = [1, 1, 1, 1, 1]
    repaired = repair_individual(ind, min_remove=2, max_remove=5)
    zeros = sum(1 for x in repaired if x == 0)
    assert zeros == 2


def test_evaluate(simple_graph):
    # Graph: S->A->B->T. Flow=5.
    # Removable nodes: ['A', 'B']
    # Individual: [1, 1] (Keep both) -> Flow 5, Cost 0

    removable_nodes = ["A", "B"]
    ind_keep_all = [1, 1]
    cache = {}
    flow, cost, coocc = evaluate(
        ind_keep_all,
        simple_graph,
        removable_nodes,
        "SuperSource",
        "SuperSink",
        shared_cache=cache,
    )

    assert flow == 5.0
    assert cost == 0

    # Individual: [0, 1] (Remove A) -> Flow 0, Cost 1
    ind_remove_A = [0, 1]
    flow, cost, coocc = evaluate(
        ind_remove_A,
        simple_graph,
        removable_nodes,
        "SuperSource",
        "SuperSink",
        shared_cache=cache,
    )

    assert flow == 0.0
    assert cost == 1


def test_pipeline_integration(simple_graph):
    # Integration smoke test for GAOptimizer
    sources = ["SuperSource"]
    sinks = ["SuperSink"]

    # The simple graph has only A, B removable

    # Fix constructor: G, sources, sinks, **kwargs
    opt = GAOptimizer(simple_graph, sources, sinks)

    # Run for 2 generations, tiny population
    # Fix return values: best_nodes, logs, pop, norm
    results, logs, pop, norm = opt.run(
        population_size=10,
        generations=2,
        early_stop_gen=10,
        use_multiprocessing=False,  # Important for testing environment without heavy setup
    )

    assert isinstance(results, dict)
    assert isinstance(logs, dict)
    assert len(logs) == 2  # 2 generations recorded
