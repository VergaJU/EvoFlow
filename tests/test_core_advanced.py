from EvoFlow.core import (
    AdaptiveRates,
    constrained_evaluate,
    select_for_diversification,
    GAOptimizer,
)
from deap import base, creator

# Ensure creator is setup
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)
    # Add crowding_dist attribute for diversification test
    creator.Individual.crowding_dist = 0.0


def test_adaptive_rates():
    rates = AdaptiveRates(
        base_cxpb=0.9,
        base_mtpb=0.3,
        base_indpb=0.1,
        min_cxpb=0.5,
        min_mtpb=0.1,
        min_indpb=0.05,
        stagnation_trigger=5,
        decay=0.9,
    )

    # 1. Test Decay
    initial_cx = rates.cxpb
    rates.step(generations_without_improvement=0)
    assert rates.cxpb < initial_cx
    assert rates.cxpb == max(0.5, initial_cx * 0.9)

    # 2. Test Stagnation / Reheat
    # Force rates to minimum
    rates.cxpb = 0.5
    rates.mtpb = 0.1
    rates.indpb = 0.05

    # Trigger reheat
    rates.step(generations_without_improvement=5)

    assert rates.burst_active is True
    # Reheated values should be higher than minimum but lower than original base (due to attenuation)
    assert rates.cxpb > 0.5
    assert rates.cxpb <= 0.9


def test_constrained_evaluate(simple_graph):
    # simple_graph has nodes: SuperSource, A, B, SuperSink
    # Removable: A, B
    removable_nodes = ["A", "B"]
    # Case 1: Feasible (Remove 1 node, Max 1)
    ind_valid = [0, 1]  # Remove A
    res_valid = constrained_evaluate(
        ind_valid,
        simple_graph,
        removable_nodes,
        "SuperSource",
        "SuperSink",
        max_remove=1,
    )
    # Should return normal fitness (Flow, Cost, Redundancy)
    # Flow=0 (A removed), Cost=1, Redundancy=0
    assert res_valid[0] == (0.0, 1, 0)

    # Case 2: Infeasible (Remove 2 nodes, Max 1)
    ind_invalid = [0, 0]  # Remove A and B
    res_invalid = constrained_evaluate(
        ind_invalid,
        simple_graph,
        removable_nodes,
        "SuperSource",
        "SuperSink",
        max_remove=1,
    )

    # Should return penalty
    # Penalty base is 1e6.
    # Distance = (2 - 1)^2 = 1.0
    # Expected = (1e6 + 1.0, 1e6 + 1.0, 1e6 + 1.0)
    assert res_invalid[0][0] > 1e5
    assert res_invalid[0] == (1000001.0, 1000001.0, 1000001.0)


def test_select_for_diversification():
    # Create dummy population where individuals are non-dominated
    # so they all end up in the first front
    pop = []

    # Ind 0: Best Flow (0, 10, 10)
    ind0 = creator.Individual([1, 1])
    ind0.fitness.values = (0.0, 10.0, 10.0)
    ind0.fitness.crowding_dist = 0.0  # Most crowded
    pop.append(ind0)

    # Ind 1: Best Node (10, 0, 10)
    ind1 = creator.Individual([1, 1])
    ind1.fitness.values = (10.0, 0.0, 10.0)
    ind1.fitness.crowding_dist = 10.0
    pop.append(ind1)

    # Ind 2: Best Redundancy (10, 10, 0)
    ind2 = creator.Individual([1, 1])
    ind2.fitness.values = (10.0, 10.0, 0.0)
    ind2.fitness.crowding_dist = 5.0
    pop.append(ind2)

    # Ind 3: Balanced (5, 5, 5) - dominated by none? No, 5<10.
    # Actually, let's just stick to the 3 extremes which are definitely non-dominated

    selected = select_for_diversification(pop, n=2)

    # Should select the 2 with SMALLEST crowding distance
    # Ind0 (0.0) and Ind2 (5.0). Ind1 is 10.0 (least crowded).

    assert len(selected) == 2

    # Verify we got Ind0 and Ind2 (order might depend on sort stability but usually by key)
    # Sorted by CD: Ind0 (0.0), Ind2 (5.0), Ind1 (10.0)
    assert selected[0].fitness.crowding_dist == 0.0
    assert selected[1].fitness.crowding_dist == 5.0


def test_ga_dynamic_mode(simple_graph):
    sources = ["SuperSource"]
    sinks = ["SuperSink"]

    # Initialize with dynamic=True
    opt = GAOptimizer(
        simple_graph,
        sources,
        sinks,
        dynamic=True,
        repair=True,  # Dynamic mode recommends repair
    )

    # Run for a few generations
    results, logs, pop, norm = opt.run(
        population_size=10,
        generations=5,
        early_stop_gen=10,
        use_multiprocessing=False,
        guess_parameters=False,
    )

    assert len(logs) == 5
    # Verify logs contain dynamic rate info
    assert "cxpb" in logs[0]
    assert "mtpb" in logs[0]


def test_ga_delta_penalty(simple_graph):
    sources = ["SuperSource"]
    sinks = ["SuperSink"]

    # Initialize with delta_penalty=True
    opt = GAOptimizer(
        simple_graph,
        sources,
        sinks,
        delta_penalty=True,
        max_remove=1,  # Tight constraint
    )

    # Run for a few generations
    results, logs, pop, norm = opt.run(
        population_size=10,
        generations=2,
        early_stop_gen=10,
        use_multiprocessing=False,
        guess_parameters=False,
    )

    assert len(logs) == 2
