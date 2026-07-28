import json

import numpy as np
from deap import base, creator

from EvoFlow import core
from EvoFlow.core import (
    AdaptiveRates,
    constrained_evaluate,
    select_for_diversification,
    GAOptimizer,
)

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


# --- DiskLogger tests ---


def test_disk_logger_basic(tmp_path):
    log_file = tmp_path / "test.jsonl"
    logger = core.DiskLogger(str(log_file), flush_every=10)
    logger.log_generation(0, {"Flow": 1.0, "Cost": 2})
    logger.log_generation(1, {"Flow": 0.5, "Cost": 1})
    logger.close()

    lines = log_file.read_text().strip().split("\n")
    assert len(lines) == 2
    rec0 = json.loads(lines[0])
    assert rec0["gen"] == 0
    assert rec0["Flow"] == 1.0
    assert rec0["Cost"] == 2


def test_disk_logger_visited_nodes_diff(tmp_path):
    log_file = tmp_path / "test.jsonl"
    logger = core.DiskLogger(str(log_file), flush_every=10)
    logger.log_generation(0, {"visited_nodes": ["A", "B", "C"]})
    logger.log_generation(1, {"visited_nodes": ["A", "B", "C", "D"]})
    logger.log_generation(2, {"visited_nodes": ["A", "B", "C", "D", "E"]})
    logger.close()

    lines = log_file.read_text().strip().split("\n")
    rec0 = json.loads(lines[0])
    rec1 = json.loads(lines[1])
    rec2 = json.loads(lines[2])

    assert rec0["visited_nodes"] == ["A", "B", "C"]
    assert rec1["visited_nodes"] == ["D"]
    assert rec2["visited_nodes"] == ["E"]


def test_disk_logger_visited_nodes_repeat(tmp_path):
    log_file = tmp_path / "test.jsonl"
    logger = core.DiskLogger(str(log_file), flush_every=10)
    logger.log_generation(0, {"visited_nodes": ["A", "B"]})
    logger.log_generation(1, {"visited_nodes": ["A", "B"]})
    logger.close()

    lines = log_file.read_text().strip().split("\n")
    rec0 = json.loads(lines[0])
    rec1 = json.loads(lines[1])
    assert rec0["visited_nodes"] == ["A", "B"]
    assert rec1["visited_nodes"] == []  # no new nodes


def test_disk_logger_to_native():
    obj = {
        "a": np.float64(1.5),
        "b": np.int32(3),
        "c": [np.float32(0.5), np.float64(2.0)],
        "d": np.array([1, 2, 3]),
    }
    native = core.DiskLogger._to_native(obj)
    assert native["a"] == 1.5
    assert isinstance(native["a"], float)
    assert native["b"] == 3
    assert isinstance(native["b"], int)
    assert native["c"] == [0.5, 2.0]
    assert native["d"] == [1, 2, 3]


def test_disk_logger_flush_behavior(tmp_path):
    log_file = tmp_path / "test.jsonl"
    logger = core.DiskLogger(str(log_file), flush_every=2)
    logger.log_generation(0, {"val": 1})
    logger.log_generation(1, {"val": 2})
    logger.log_generation(2, {"val": 3})
    logger.close()

    lines = log_file.read_text().strip().split("\n")
    assert len(lines) == 3


def test_load_logs(tmp_path):
    log_file = tmp_path / "test.jsonl"
    with open(log_file, "w") as f:
        f.write(json.dumps({"gen": 0, "Flow": 1.0, "visited_nodes": ["A", "B"]}) + "\n")
        f.write(json.dumps({"gen": 1, "Flow": 0.5, "visited_nodes": ["C"]}) + "\n")

    logs = core.load_logs(str(log_file))
    assert 0 in logs
    assert 1 in logs
    assert logs[0]["Flow"] == 1.0
    assert logs[0]["visited_nodes"] == ["A", "B"]
    assert logs[1]["visited_nodes"] == ["A", "B", "C"]


def test_load_logs_no_visited(tmp_path):
    log_file = tmp_path / "test.jsonl"
    with open(log_file, "w") as f:
        f.write(json.dumps({"gen": 0, "Flow": 1.0}) + "\n")

    logs = core.load_logs(str(log_file))
    assert "visited_nodes" not in logs[0]


def test_adaptive_rates_reheat_multiple_times():
    rates = AdaptiveRates(
        base_cxpb=0.9,
        base_mtpb=0.3,
        base_indpb=0.1,
        min_cxpb=0.5,
        min_mtpb=0.1,
        min_indpb=0.05,
        stagnation_trigger=3,
        decay=0.5,
        reheat_attenuation=0.5,
    )
    rates.cxpb = 0.5
    rates.mtpb = 0.1
    rates.indpb = 0.05

    rates.step(generations_without_improvement=3)
    first_peak = rates.cxpb

    rates.cxpb = 0.5
    rates.mtpb = 0.1
    rates.indpb = 0.05
    rates.step(generations_without_improvement=3)
    second_peak = rates.cxpb

    assert first_peak > 0.5
    assert second_peak > 0.5
    assert second_peak < first_peak


def test_adaptive_rates_no_stagnation_decay():
    rates = AdaptiveRates(
        base_cxpb=0.9,
        base_mtpb=0.3,
        base_indpb=0.1,
        decay=0.5,
        stagnation_trigger=10,
    )
    initial_cx = rates.cxpb
    rates.step(generations_without_improvement=0)
    assert rates.cxpb < initial_cx
    assert rates.burst_active is False
