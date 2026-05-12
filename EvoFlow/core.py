import random
import time
import networkx as nx
import multiprocessing
from deap import base, creator, tools
from deap.benchmarks.tools import hypervolume
import numpy as np
from scipy.spatial.distance import pdist

# --- Helper Functions (Graph Algorithms) ---


def find_scc(G):
    """
    Find strongly connected components in a directed graph.
    Returns a list of sets, where each set contains nodes in a component (len > 1).
    """
    sccs = [i for i in nx.strongly_connected_components(G) if len(i) > 1]
    return sccs


def scc_redundancy(sccs):
    redundancy = 0
    for scc in sccs:
        if len(scc) > 1:
            redundancy += len(scc) ** 2
    return redundancy


def calculate_flow_capacity(G, ssource, ssink):
    """
    Calculates max flow capacity between source and sink.
    """
    flow_dict = {}
    try:
        flow_value, flow_dict = nx.maximum_flow(G, ssource, ssink, capacity="capacity")
    except Exception as e:
        # Fallback or error handling if graph structure is invalid
        print(f"Error calculating flow: {e}")
        flow_value = 0
    return flow_value, flow_dict


def extract_active_subgraph(G, source="SuperSource", sink="SuperSink"):
    """
    Extracts the subgraph containing all paths between source and sink.
    """
    if source not in G or sink not in G:
        print(f"Source {source} or Sink {sink} not in graph")
        return nx.DiGraph()

    reachable_from_source_nodes = nx.descendants(G, source)
    reachable_to_sink_nodes = nx.descendants(G.reverse(), sink)

    nodes_in_path = reachable_from_source_nodes & reachable_to_sink_nodes
    nodes_to_keep = nodes_in_path | {source, sink}

    return G.subgraph(nodes_to_keep).copy()


# --- GA Individual Management ---


def generate_individual(icls, removable_nodes, max_remove=3):
    """
    Generates a random individual (binary mask).
    1 = Keep Node, 0 = Remove Node.
    """
    ind = [1] * len(removable_nodes)
    n_remove = random.randint(1, max_remove)
    # Ensure we don't try to remove more than exist
    n_remove = min(n_remove, len(removable_nodes))

    to_remove = random.sample(range(len(removable_nodes)), n_remove)
    for i in to_remove:
        ind[i] = 0
    return icls(ind)


def repair_individual(ind, min_remove=2, max_remove=10):
    """
    Repairs an individual to stay within deletion bounds.
    """
    current_removed = ind.count(0)
    modified = False

    # Too many deletions (flip 0 -> 1)
    if current_removed > max_remove:
        zero_indices = [i for i, g in enumerate(ind) if g == 0]
        to_flip = random.sample(zero_indices, current_removed - max_remove)
        for i in to_flip:
            ind[i] = 1
        modified = True

    # Too few deletions (flip 1 -> 0)
    elif current_removed < min_remove:
        one_indices = [i for i, g in enumerate(ind) if g == 1]
        if len(one_indices) >= (min_remove - current_removed):
            to_flip = random.sample(one_indices, min_remove - current_removed)
            for i in to_flip:
                ind[i] = 0
        modified = True

    # If we modified the genes, we MUST invalidate the fitness
    if modified and hasattr(ind, "fitness"):
        del ind.fitness.values

    return ind


def mutate(individual, indpb=0.05):
    """
    Wrapper for shuffle mutation that includes cleanup.
    """
    tools.mutFlipBit(individual, indpb)
    # cleanup_individual(individual, min_remove, max_remove)
    return (individual,)


# --- Util get nodes --


def get_nodes(removable_nodes, individual):
    return [node for i, node in enumerate(removable_nodes) if individual[i] == 0]


# --- Evaluation Function (Must be top-level for multiprocessing) ---


def evaluate(individual, G, removable_nodes, ssource, ssink, shared_cache):
    """
    Fitness Function.
    Objectives (minimized):
    1. Flow Value (Max Flow)
    2. Number of Removed Nodes
    3. Co-occurrence in SCCs
    """
    # Identify nodes to remove
    removed_nodes_names = set(get_nodes(removable_nodes, individual))

    # --- CACHE CHECK ---
    cache_key = frozenset(removed_nodes_names)
    if cache_key in shared_cache:
        return shared_cache[cache_key], {}  # Returns all 3 objectives instantly!

    # --- GRAPH CREATION ---
    nodes_to_keep = [n for n in G.nodes if n not in removed_nodes_names]
    G_sub = G.subgraph(nodes_to_keep)

    # --- EVALUATION ---
    # 1. Flow Capacity
    flow_dict = {}
    try:
        flow_value, flow_dict = nx.maximum_flow(
            G_sub, ssource, ssink, capacity="capacity"
        )
    except Exception:
        flow_value = 0

    # 2. SCC Co-occurrence
    sccs = find_scc(G_sub)
    redundancy = scc_redundancy(sccs=sccs)

    # 3. Cost (Number of nodes removed)
    cost = len(removed_nodes_names)

    # --- SAVE TO CACHE ---
    result = (flow_value, cost, redundancy)
    shared_cache[cache_key] = result

    return result, flow_dict


# --- Constraint Helpers ---


def check_feasible(individual, max_remove):
    """Returns True if the individual removes no more than max_remove nodes."""
    return individual.count(0) <= max_remove


def penalty_distance(individual, max_remove):
    """Calculates how far the individual is from being feasible."""
    return float((individual.count(0) - max_remove) ** 2)


def constrained_evaluate(
    individual, G, removable_nodes, ssource, ssink, max_remove, shared_cache
):
    """
    A manual wrapper that mimics DeltaPenalty but is picklable.
    """
    if not check_feasible(individual, max_remove):
        # Penalty values (must be a tuple for multi-objective)
        penalty_base = (1e6, 1e6, 1e6)
        dist = penalty_distance(individual, max_remove)
        return tuple(p + dist for p in penalty_base), {}

    # Otherwise, return actual fitness
    return evaluate(individual, G, removable_nodes, ssource, ssink, shared_cache)


# --- Genotypic diversity ----


def genotypic_diversity(population):
    if len(population) < 2:
        return 0.0
    pop_array = np.array(population, dtype=int)

    distances = pdist(pop_array, metric="hamming")
    return np.mean(distances)


# --- Get visited nodes per generation ---


def collapse_pop(population):
    pop_array = np.array(population, dtype=int)
    # Collapse: Result is 1 only if ALL are 1. If any are 0, result is 0.
    collapsed = np.all(pop_array, axis=0).astype(int)
    return collapsed


# --- Select candidates ---


def select_candidates(population, removable_nodes):
    front = tools.sortNondominated(population, len(population), first_front_only=True)[
        0
    ]
    fitness_matrix = np.array([ind.fitness.values for ind in front])
    mins = fitness_matrix.min(axis=0)
    maxs = fitness_matrix.max(axis=0)
    norm = (fitness_matrix - mins) / (
        maxs - mins + 1e-12
    )  # Tiny epsilon to avoid zero divisions
    distances = np.linalg.norm(
        norm, axis=1
    )  # euclidean distance from best scenarion (origin)
    best_index = np.argmin(
        distances
    )  # get index "best" individual, the closest to the origin
    best_balanced = front[best_index]  # "Best" individual

    results = {
        "Best_balanced": best_balanced,
        "Best_Flow": min(
            front, key=lambda ind: ind.fitness.values[0]
        ),  # get best individual for Flow reduction
        "Best_Cost": min(
            front, key=lambda ind: ind.fitness.values[1]
        ),  # Get individual with least removed nodes
        "Best_Redundancy": min(
            front, key=lambda ind: ind.fitness.values[2]
        ),  # Get individual for Redundancy reduction
    }

    # Format into a serializable dictionary
    fitness_keys = ["Flow", "Node", "Redundancy"]
    output = {}
    for (
        key,
        ind,
    ) in (
        results.items()
    ):  # populate results with fitnesses, number of removed nodes, and removed nodes
        output[key] = {
            "fitness": {
                k: float(v) for k, v in zip(fitness_keys, ind.fitness.values)
            },  # Convert to plain floats
            "num_removed": int(ind.count(0)),
            "removed_nodes": get_nodes(removable_nodes=removable_nodes, individual=ind),
        }
    return output, norm


# --- Select crowded individuals to burst rates ---


def select_for_diversification(population, n):
    """Select individuals with LOWEST crowding distance — most crowded = least diverse."""
    front = tools.sortNondominated(population, len(population), first_front_only=True)[
        0
    ]
    # Sort by crowding distance ascending (most crowded first)
    sorted_front = sorted(front, key=lambda ind: ind.fitness.crowding_dist)
    return sorted_front[:n]  # Return the most crowded ones to mutate


# --- Create class to handle adaptive rates ---


class AdaptiveRates:
    def __init__(
        self,
        base_cxpb,
        base_mtpb,
        base_indpb,
        min_cxpb=0.5,
        min_mtpb=0.1,
        min_indpb=None,
        max_cxpb=0.95,
        max_mtpb=0.8,
        max_indpb=None,
        decay=0.997,
        stagnation_trigger=3,
        reheat_attenuation=0.9,
    ):  # <--- New: how much the "spike" weakens each time
        self.base_cxpb = base_cxpb
        self.base_mtpb = base_mtpb
        self.base_indpb = base_indpb

        self.min_cxpb = min_cxpb
        self.min_mtpb = min_mtpb
        self.min_indpb = min_indpb or base_indpb * 0.5

        self.max_cxpb = max_cxpb
        self.max_mtpb = max_mtpb
        self.max_indpb = max_indpb or min(base_indpb * 5, 0.5)

        self.decay = decay
        self.stagnation_trigger = stagnation_trigger
        self.reheat_attenuation = reheat_attenuation

        # This keeps track of our current "ceiling"
        self.current_reheat_factor = 1.0

        self.cxpb = base_cxpb
        self.mtpb = base_mtpb
        self.indpb = base_indpb

        self.burst_active = False

    def step(self, generations_without_improvement):
        at_minimum = (
            abs(self.cxpb - self.min_cxpb) < 1e-4
            and abs(self.mtpb - self.min_mtpb) < 1e-4
            and abs(self.indpb - self.min_indpb) < 1e-4
        )

        if at_minimum and generations_without_improvement >= self.stagnation_trigger:
            self.reheat()
            self.burst_active = True
        else:
            self.burst_active = False
            # Normal exponential decay toward minimums
            self.cxpb = max(self.min_cxpb, self.cxpb * self.decay)
            self.mtpb = max(self.min_mtpb, self.mtpb * self.decay)
            self.indpb = max(self.min_indpb, self.indpb * self.decay)

    def reheat(self):
        """
        Implementation of Adaptive Reheating:
        Every time we reheat, the peak is lower than the previous peak.
        """
        # 1. Lower the 'energy' for this next cycle
        self.current_reheat_factor *= self.reheat_attenuation

        # 2. Calculate the new peak relative to the range [Min, Base]
        # This ensures we always stay above the minimum but spike less each time.
        self.cxpb = (
            self.min_cxpb
            + (self.base_cxpb - self.min_cxpb) * self.current_reheat_factor
        )
        self.mtpb = (
            self.min_mtpb
            + (self.base_mtpb - self.min_mtpb) * self.current_reheat_factor
        )
        self.indpb = (
            self.min_indpb
            + (self.base_indpb - self.min_indpb) * self.current_reheat_factor
        )


# --- MockIndividual to normalize front ---
class MockIndividual:
    def __init__(self, values):
        # We create a fake fitness object with a values attribute
        self.fitness = type("Fitness", (), {"values": values})


# --- GA Engine Class ---


class GAOptimizer:
    def __init__(
        self,
        G,
        sources,
        sinks,
        max_remove=10,
        min_remove=1,
        delta_penalty=False,
        repair=False,
        norm_hypervolume=False,
        dynamic=False,
        base_cxpb=0.75,  # Start mating 75% of the population
        base_mtpb=0.9,  # Start mutating 90% of the population
        base_indpb=5,  # Start with a probability of mutating 5 bits per muated individual
        use_multiprocessing=True,
        verbose=False,
        **kwargs,
    ):
        self.G = G
        self.SG = extract_active_subgraph(G)
        self.sources = sources
        self.sinks = sinks
        self.min_remove = min_remove
        self.max_remove = max_remove
        self.delta_penalty = delta_penalty
        self.repair = repair
        self.dynamic = dynamic
        self.norm_hypervolume = norm_hypervolume
        self.use_multiprocessing = (
            use_multiprocessing  # FIX 1: Fixed typo 'use_multiptocessing'
        )
        self.verbose = verbose

        if self.dynamic and not self.repair:
            print(
                "\nWARNING: Dynamic mutation rates will introduce a higher number of removed nodes. "
                "If repair=True is not set, it will lead to several individuals not respecting constraints.\n"
            )

        # FIX 2: Cleaner set subtraction for determining removable nodes
        self.removable_nodes = list(set(self.SG.nodes) - set(sources) - set(sinks))
        self.N = len(self.removable_nodes)

        # Calculate Baselines for Normalization
        # (We pass a standard empty dict here because the baseline only evaluates 1 individual)
        (self.base_flow, _, self.base_redundancy), _ = evaluate(
            [1] * self.N,  # Individual where all nodes are kept (all 1s)
            self.G,
            self.removable_nodes,
            "SuperSource",
            "SuperSink",
            shared_cache={},
        )

        # Handle cases where base values might be 0 to avoid DivisionByZero
        self.base_flow = max(self.base_flow, 1.0)
        self.base_cost = max(self.max_remove, 1.0)
        self.base_redundancy = max(self.base_redundancy, 1.0)

        self._norm_denominator = np.array(
            [self.base_flow, self.base_cost, self.base_redundancy]
        )

        # Define initial rates
        self.base_cxpb = base_cxpb
        self.base_mtpb = base_mtpb
        self.base_indpb = base_indpb / self.N

        # DEAP Setup
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()

        # Registration
        self.toolbox.register(
            "individual",
            generate_individual,
            icls=creator.Individual,
            removable_nodes=self.removable_nodes,
            max_remove=self.max_remove,
        )
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )
        self.toolbox.register("mate", tools.cxUniform, indpb=0.5)
        self.toolbox.register("mutate", tools.mutFlipBit)
        self.toolbox.register(
            "repair",
            repair_individual,
            min_remove=self.min_remove,
            max_remove=self.max_remove,
        )
        self.toolbox.register("select", tools.selNSGA2)

        # NOTE: map and evaluate are now registered dynamically inside run()!

    def run(
        self,
        population_size=300,
        generations=100,
        early_stop_gen=10,
        tol=1e-6,
        verbose=False,
        **kwargs,
    ):
        """
        Runs the Genetic Algorithm.
        """
        stagnation_trigger = early_stop_gen // 10
        rates = AdaptiveRates(
            base_cxpb=self.base_cxpb,
            base_mtpb=self.base_mtpb,
            base_indpb=self.base_indpb,
            stagnation_trigger=stagnation_trigger,
        )

        # FIX 3: Initialize pool and manager variables in the local scope
        # so the `finally` block doesn't crash on UnboundLocalError
        pool = None
        manager = None
        shared_cache = {}

        if self.use_multiprocessing:
            manager = multiprocessing.Manager()
            shared_cache = manager.dict()  # Real inter-process shared memory
            pool = multiprocessing.Pool()
            self.toolbox.register("map", pool.map)
        else:
            self.toolbox.register("map", map)

        # FIX 4: Dynamically register evaluate to inject the proper shared_cache
        if self.delta_penalty:
            self.toolbox.register(
                "evaluate",
                constrained_evaluate,
                G=self.SG,
                removable_nodes=self.removable_nodes,
                ssource="SuperSource",
                ssink="SuperSink",
                max_remove=self.max_remove,
                shared_cache=shared_cache,
            )
        else:
            self.toolbox.register(
                "evaluate",
                evaluate,
                G=self.SG,
                removable_nodes=self.removable_nodes,
                ssource="SuperSource",
                ssink="SuperSink",
                shared_cache=shared_cache,
            )

        try:
            population = self.toolbox.population(n=population_size)
            if self.verbose:
                print(f"{time.ctime()} - Evaluating fitness for initial population...")

            # Evaluate initial population
            invalid_ind = [ind for ind in population if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, (fit, flow) in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            # for ind, fit in zip(invalid_ind, fitnesses):
            #     ind.fitness.values = fit

            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("min", min)

            ref_point = (1.1, 1.1, 1.1)
            best_hv = 0.0
            generations_without_improvement = 0
            logs = {}
            if self.verbose:
                print(
                    f"{time.ctime()} - Starting evolution for {generations} generations"
                )

            for gen in range(generations):
                # Selection & Cloning
                offspring = self.toolbox.select(population, len(population))
                offspring = list(map(self.toolbox.clone, offspring))

                # Crossover
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < rates.cxpb:
                        self.toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values

                # Mutation
                for mutant in offspring:
                    if random.random() < rates.mtpb:
                        self.toolbox.mutate(mutant, indpb=rates.indpb)
                        del mutant.fitness.values

                # Dynamic Burst Mutation
                if self.dynamic and rates.burst_active:
                    crowded = select_for_diversification(
                        population=population, n=len(population) // 5
                    )
                    for ind in crowded:
                        clone = self.toolbox.clone(ind)
                        self.toolbox.mutate(clone, indpb=rates.indpb)
                        del clone.fitness.values
                        # FIX 5: Removed duplicate self.toolbox.repair(clone) here,
                        # because the repair block below catches all offspring anyway.
                        offspring.append(clone)

                # Repair
                if self.repair:
                    for mutant in offspring:
                        self.toolbox.repair(mutant)

                # Evaluation
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)

                for ind, (fit, flow) in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                # Replacement (Elitism via NSGA2)
                population[:] = self.toolbox.select(
                    population + offspring, len(population)
                )

                # FIX 6: Initialize metric trackers BEFORE the if-statement
                # to prevent UnboundLocalError when feasible_pop is empty
                avg_cd = 0.0
                current_hv = 0.0

                feasible_pop = [
                    ind for ind in population if ind.fitness.values[0] < 1e5
                ]

                if feasible_pop:
                    front = tools.sortNondominated(
                        feasible_pop, len(feasible_pop), first_front_only=True
                    )[0].copy()

                    crowding_distances = np.array(
                        [ind.fitness.crowding_dist for ind in front], dtype=float
                    )

                    finite_cd = crowding_distances[np.isfinite(crowding_distances)]
                    if len(finite_cd) > 0:
                        avg_cd = np.mean(finite_cd)

                    if self.norm_hypervolume:
                        fitness_matrix = np.array([ind.fitness.values for ind in front])
                        norm_matrix = fitness_matrix / self._norm_denominator
                        normalized_mock_front = []
                        for norm_values in norm_matrix:
                            mock_ind = type("MockIndividual", (), {})()
                            mock_ind.fitness = creator.FitnessMin()
                            mock_ind.fitness.values = tuple(norm_values)
                            normalized_mock_front.append(mock_ind)

                        current_hv = hypervolume(normalized_mock_front, ref_point)
                    else:
                        current_hv = hypervolume(front, self._norm_denominator)

                    if current_hv > best_hv + tol:
                        best_hv = current_hv
                        generations_without_improvement = 0
                    else:
                        generations_without_improvement += 1

                if self.dynamic:
                    rates.step(generations_without_improvement)

                # --- STATS AND LOGGING ---
                fitness_keys = ["Flow", "Node", "Redundancy"]

                # `stats.compile` returns {"min":[flow, cost, red]}
                # We unpack it into our custom keys
                stat_values = stats.compile(population)["min"]
                record = {k: v for k, v in zip(fitness_keys, stat_values)}

                record["hv"] = current_hv
                record["cd"] = avg_cd
                record["hd"] = genotypic_diversity(population=population)
                record["visited_nodes"] = get_nodes(
                    removable_nodes=self.removable_nodes,
                    individual=collapse_pop(population=population),
                )
                record["cxpb"] = rates.cxpb
                record["mtpb"] = rates.mtpb
                record["indpb"] = rates.indpb
                logs[gen] = record

                if self.verbose:
                    # FIX 7: Fixed KeyError. record["min"] doesn't exist anymore!
                    print(
                        f"{time.ctime()} - Gen {gen}: "
                        f"Flow: {record['Flow']}, Nodes: {record['Node']}, SCCs: {record['Redundancy']}. "
                        f"HV: {round(record['hv'], 3)}"
                    )

                if generations_without_improvement >= early_stop_gen:
                    if self.verbose:
                        print(
                            f"{time.ctime()} - Early stopping at generation {gen} (No HV improvement)."
                        )
                    break

        finally:
            # FIX 8: Safe cleanup to prevent memory leaks or UnboundLocalError
            if self.use_multiprocessing:
                if pool is not None:
                    pool.close()
                    pool.join()
                if manager is not None:
                    manager.shutdown()

        results, norm = select_candidates(
            population=population, removable_nodes=self.removable_nodes
        )

        return results, logs, population, norm
