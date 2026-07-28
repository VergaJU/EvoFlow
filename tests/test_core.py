from unittest.mock import MagicMock

import networkx as nx
import numpy as np
import pytest
from array import array
from deap import base, creator

from EvoFlow import core
from EvoFlow.core import (
    extract_active_subgraph,
    generate_individual,
    repair_individual,
    evaluate,
    GAOptimizer,
)

SCALE = 1000


def _build_csr_from_graph(G):
    node_order = list(G.nodes)
    node_to_idx = {n: i for i, n in enumerate(node_order)}
    G_scaled = G.copy()
    for u, v, d in G_scaled.edges(data=True):
        d["capacity"] = int(round(d.get("capacity", 1.0) * SCALE))
    csr = nx.to_scipy_sparse_array(
        G_scaled, nodelist=node_order, weight="capacity", format="csr", dtype=np.int64
    )
    removable_nodes = [n for n in node_order if n not in ("SuperSource", "SuperSink")]
    removable_csr_indices = np.array(
        [node_to_idx[n] for n in removable_nodes], dtype=np.int64
    )
    return (
        csr,
        removable_csr_indices,
        removable_nodes,
        node_to_idx["SuperSource"],
        node_to_idx["SuperSink"],
    )


def _set_worker_globals(csr, removable_csr_indices, source_idx, sink_idx):
    core._worker_csr = csr
    core._worker_removable_csr_indices = removable_csr_indices
    core._worker_source_idx = source_idx
    core._worker_sink_idx = sink_idx


# Ensure creator is setup if running tests in isolation
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0))
if not hasattr(creator, "Individual"):
    creator.create("Individual", array, typecode="B", fitness=creator.FitnessMin)


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


def _simple_csr_graph():
    G = nx.DiGraph()
    G.add_node("SuperSource")
    G.add_node("A", percolation=1.0)
    G.add_node("B", percolation=0.5)
    G.add_node("SuperSink")
    G.add_edge("SuperSource", "A", capacity=10.0, weight=1.0)
    G.add_edge("A", "B", capacity=5.0, weight=1.0)
    G.add_edge("B", "SuperSink", capacity=10.0, weight=1.0)
    return _build_csr_from_graph(G)


def test_evaluate_keep_all():
    csr, ridx, removable, src_idx, sink_idx = _simple_csr_graph()
    _set_worker_globals(csr, ridx, src_idx, sink_idx)

    ind = [1, 1]  # keep both A and B
    (flow, cost, red), _ = evaluate(ind, scale=SCALE, max_remove=10, constrained=False)
    # Bottleneck A->B = 5.0 (scaled to 5000, /1000)
    assert flow == 5.0
    assert cost == 0


def test_evaluate_remove_node():
    csr, ridx, removable, src_idx, sink_idx = _simple_csr_graph()
    _set_worker_globals(csr, ridx, src_idx, sink_idx)

    ind = [0, 1]  # remove A
    (flow, cost, red), _ = evaluate(ind, scale=SCALE, max_remove=10, constrained=False)
    assert flow == 0.0
    assert cost == 1


def test_evaluate_constrained_penalty():
    csr, ridx, removable, src_idx, sink_idx = _simple_csr_graph()
    _set_worker_globals(csr, ridx, src_idx, sink_idx)

    ind = [0, 0]  # remove A and B (2 removals, max=1)
    (f0, f1, f2), _ = evaluate(ind, scale=SCALE, max_remove=1, constrained=True)
    assert f0 > 1e5


def test_evaluate_constrained_feasible():
    csr, ridx, removable, src_idx, sink_idx = _simple_csr_graph()
    _set_worker_globals(csr, ridx, src_idx, sink_idx)

    ind = [0, 1]  # remove A only (1 removal, max=1)
    (flow, cost, red), _ = evaluate(ind, scale=SCALE, max_remove=1, constrained=True)
    assert flow == 0.0
    assert cost == 1


def test_evaluate_scc_redundancy():
    """Build a graph with a cycle so SCC redundancy > 0."""
    G = nx.DiGraph()
    G.add_node("SuperSource")
    G.add_node("A")
    G.add_node("B")
    G.add_node("C")
    G.add_node("SuperSink")
    G.add_edge("SuperSource", "A", capacity=10.0)
    G.add_edge("A", "B", capacity=10.0)
    G.add_edge("B", "C", capacity=10.0)
    G.add_edge("C", "A", capacity=10.0)  # cycle A->B->C->A
    G.add_edge("A", "SuperSink", capacity=10.0)
    csr, ridx, removable, src_idx, sink_idx = _build_csr_from_graph(G)
    _set_worker_globals(csr, ridx, src_idx, sink_idx)

    # 3 removable nodes: A, B, C
    ind = [1, 1, 1]  # keep all
    (flow, cost, red), _ = evaluate(ind, scale=SCALE, max_remove=10, constrained=False)
    # The 3-cycle {A,B,C} should give redundancy = 3^2 = 9
    assert red == 9


def test_pipeline_integration(simple_graph):
    sources = ["SuperSource"]
    sinks = ["SuperSink"]
    opt = GAOptimizer(simple_graph, sources, sinks)
    results, logs, pop, norm = opt.run(
        population_size=10,
        generations=2,
        early_stop_gen=10,
        guess_parameters=False,
        use_multiprocessing=False,
    )
    assert isinstance(results, dict)
    assert isinstance(logs, dict)
    assert len(logs) == 2


def test_guess_parameters_direct(complex_graph):
    sources = ["SuperSource"]
    sinks = ["SuperSink"]
    opt = GAOptimizer(complex_graph, sources, sinks)
    population, patience = opt.guess_parameters_gam()
    assert isinstance(population, (int, float, np.integer))
    assert isinstance(patience, (int, float, np.integer))
    assert population > 0
    assert patience > 0


def test_get_nodes():
    removable = ["X", "Y", "Z"]
    ind = [0, 1, 0]
    result = core.get_nodes(removable, ind)
    assert result == ["X", "Z"]


def test_get_nodes_none():
    removable = ["X", "Y", "Z"]
    ind = [1, 1, 1]
    assert core.get_nodes(removable, ind) == []


def test_mutate():
    ind = creator.Individual([1, 1, 1, 1, 1])
    (mutated,) = core.mutate(ind, indpb=1.0)
    assert mutated is ind
    assert any(g == 0 for g in mutated)


def test_mutate_low_prob():
    ind = creator.Individual([1, 1, 1, 1, 1])
    (mutated,) = core.mutate(ind, indpb=0.0)
    assert all(g == 1 for g in mutated)


def test__init_worker(simple_graph):
    csr, ridx, removable, src_idx, sink_idx = _build_csr_from_graph(simple_graph)
    core._init_worker(csr, ridx, src_idx, sink_idx)
    assert core._worker_csr is csr
    assert np.array_equal(core._worker_removable_csr_indices, ridx)
    assert core._worker_source_idx == src_idx
    assert core._worker_sink_idx == sink_idx


def test_genotypic_diversity():
    pop = [np.array([1, 0, 1]), np.array([1, 1, 0])]
    div = core.genotypic_diversity(pop)
    assert div == pytest.approx(2 / 3)


def test_genotypic_diversity_identical():
    pop = [np.array([1, 0, 1]), np.array([1, 0, 1])]
    assert core.genotypic_diversity(pop) == 0.0


def test_genotypic_diversity_single():
    assert core.genotypic_diversity([np.array([1, 0, 1])]) == 0.0


def test_collapse_pop():
    pop = [np.array([1, 0, 1]), np.array([1, 0, 0])]
    collapsed = core.collapse_pop(pop)
    assert list(collapsed) == [1, 0, 0]


def test_collapse_pop_all_ones():
    pop = [np.array([1, 1, 1]), np.array([1, 1, 1])]
    assert list(core.collapse_pop(pop)) == [1, 1, 1]


def test_collapse_pop_single():
    pop = [np.array([1, 0, 1])]
    assert list(core.collapse_pop(pop)) == [1, 0, 1]


def test_mock_individual():
    ind = core.MockIndividual((1.0, 2.0, 3.0))
    assert ind.fitness.values == (1.0, 2.0, 3.0)


def _graph_with_removable_node():
    G = nx.DiGraph()
    G.add_node("SuperSource")
    G.add_node("M", percolation=1.0)
    G.add_node("SuperSink")
    G.add_edge("SuperSource", "M", capacity=1.0)
    G.add_edge("M", "SuperSink", capacity=1.0)
    return G


def test__get_n_jobs_default():
    sources = ["SuperSource"]
    sinks = ["SuperSink"]
    G = _graph_with_removable_node()
    opt = GAOptimizer(G, sources, sinks)
    n = opt._get_n_jobs()
    assert n >= 1


def test__get_n_jobs_custom():
    sources = ["SuperSource"]
    sinks = ["SuperSink"]
    G = _graph_with_removable_node()
    opt = GAOptimizer(G, sources, sinks, n_jobs=1)
    assert opt._get_n_jobs() == 1


def test__get_n_jobs_invalid():
    sources = ["SuperSource"]
    sinks = ["SuperSink"]
    G = _graph_with_removable_node()
    opt = GAOptimizer(G, sources, sinks, n_jobs=0)
    with pytest.raises(ValueError):
        opt._get_n_jobs()


# --- SmartCacheMap tests ---


def test_smart_cache_hit():
    eval_func = MagicMock(return_value=((1.0, 2.0, 3.0), {}))
    cache = {tuple([1, 0, 1]): ((0.0, 0.0, 0.0), {})}
    scm = core.SmartCacheMap(evaluation_func=eval_func, cache=cache, pool=None)

    pop = [array("B", [1, 0, 1])]
    results = scm(None, pop)
    assert results[0] == ((0.0, 0.0, 0.0), {})
    eval_func.assert_not_called()


def test_smart_cache_miss():
    eval_func = MagicMock(return_value=((1.0, 2.0, 3.0), {}))
    scm = core.SmartCacheMap(evaluation_func=eval_func, cache={}, pool=None)

    pop = [array("B", [1, 0, 1])]
    results = scm(None, pop)
    assert results[0] == ((1.0, 2.0, 3.0), {})
    eval_func.assert_called_once_with(array("B", [1, 0, 1]))


def test_smart_cache_mixed():
    eval_func = MagicMock(return_value=((1.0, 2.0, 3.0), {}))
    cache = {tuple([1, 0, 1]): ((9.0, 9.0, 9.0), {})}
    scm = core.SmartCacheMap(evaluation_func=eval_func, cache=cache, pool=None)

    pop = [array("B", [1, 0, 1]), array("B", [0, 1, 0])]
    results = scm(None, pop)
    assert results[0] == ((9.0, 9.0, 9.0), {})  # cache hit
    assert results[1] == ((1.0, 2.0, 3.0), {})  # cache miss
    eval_func.assert_called_once()


def test_smart_cache_updates_cache():
    eval_func = MagicMock(return_value=((1.0, 2.0, 3.0), {}))
    cache = {}
    scm = core.SmartCacheMap(evaluation_func=eval_func, cache=cache, pool=None)

    pop = [array("B", [1, 0, 1])]
    scm(None, pop)
    assert tuple([1, 0, 1]) in cache
    assert cache[tuple([1, 0, 1])] == ((1.0, 2.0, 3.0), {})


def test_smart_cache_with_pool():
    fake_pool = MagicMock()
    fake_pool.map = MagicMock(return_value=[((2.0, 3.0, 4.0), {})])

    eval_func = MagicMock(return_value=((2.0, 3.0, 4.0), {}))
    cache = {}
    scm = core.SmartCacheMap(evaluation_func=eval_func, cache=cache, pool=fake_pool)

    pop = [array("B", [0, 0, 1])]
    results = scm(None, pop)
    assert results[0] == ((2.0, 3.0, 4.0), {})
    fake_pool.map.assert_called_once_with(eval_func, [array("B", [0, 0, 1])])


# --- select_candidates tests ---


def test_select_candidates():
    pop = [
        creator.Individual([1, 0, 1, 0]),
        creator.Individual([0, 1, 0, 1]),
        creator.Individual([1, 1, 0, 0]),
    ]
    pop[0].fitness.values = (1.0, 2.0, 3.0)
    pop[1].fitness.values = (4.0, 5.0, 6.0)
    pop[2].fitness.values = (7.0, 8.0, 9.0)
    removable = ["A", "B", "C", "D"]

    output, norm = core.select_candidates(pop, removable)
    assert set(output.keys()) == {
        "Best_balanced",
        "Best_Flow",
        "Best_Cost",
        "Best_Redundancy",
    }
    assert isinstance(norm, np.ndarray)
    for key in output:
        assert "fitness" in output[key]
        assert "num_removed" in output[key]
        assert "removed_nodes" in output[key]


def test_select_candidates_single():
    ind = creator.Individual([1, 0, 1])
    ind.fitness.values = (1.0, 2.0, 3.0)
    output, norm = core.select_candidates([ind], ["A", "B", "C"])
    assert len(output) == 4
    assert output["Best_balanced"]["removed_nodes"] == ["B"]
