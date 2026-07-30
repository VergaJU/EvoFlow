import random
import time
import networkx as nx
import multiprocessing
from array import array
from deap import base, creator, tools
from deap.benchmarks.tools import hypervolume
import numpy as np
from scipy.spatial.distance import pdist
import pandas as pd
from EvoFlow.features_extraction import NetworkFeatures
from EvoFlow.model import GAM
import math
import functools
import os
import json
from scipy.sparse.csgraph import maximum_flow, connected_components


_worker_csr = None
_worker_idx_to_node = None
_worker_removable_csr_indices = None
_worker_source_idx = None
_worker_sink_idx = None


def _init_worker(csr, removable_csr_indices, source_idx, sink_idx, node_to_idx):
    global \
        _worker_csr, \
        _worker_removable_csr_indices, \
        _worker_source_idx, \
        _worker_sink_idx, \
        _worker_idx_to_node
    _worker_csr = csr
    _worker_removable_csr_indices = removable_csr_indices
    _worker_source_idx = source_idx
    _worker_sink_idx = sink_idx
    _worker_idx_to_node = {v: k for k, v in node_to_idx.items()}


# --- Helper Functions (Graph Algorithms) ---


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
    n_remove = min(random.randint(1, max_remove), len(removable_nodes))

    to_remove = random.sample(range(len(removable_nodes)), n_remove)
    for i in to_remove:
        ind[i] = 0
    return icls(array("B", ind))


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


def mask_csr(individual):
    removed_mask = ~np.array(individual, dtype=bool)
    removed_csr_idx = _worker_removable_csr_indices[removed_mask]

    csr_masked = _worker_csr.copy()
    csr_masked[removed_csr_idx, :] = 0
    csr_masked[:, removed_csr_idx] = 0
    csr_masked.eliminate_zeros()
    return csr_masked, removed_mask


def scipy_to_flow_dict_active(flow_matrix, csr_masked):
    """
    Converts MaximumFlowResult.flow to a dict of only REAL, active
    (positive) flows on edges that exist in the original capacity
    matrix -- matching nx.maximum_flow's semantics.

    flow_matrix includes scipy's internally-added reverse/residual
    edges (with zero or negative bookkeeping values); these must be
    filtered out using the original capacity structure, not inferred
    from flow_matrix's own sparsity pattern.
    """
    # Binary mask of the ORIGINAL (real) edges only
    real_edge_mask = csr_masked.copy()
    real_edge_mask.data[:] = 1

    # Keep flow_matrix values only where a real edge exists.
    # .multiply() takes the intersection of sparsity patterns.
    filtered = flow_matrix.multiply(real_edge_mask).tocsr()

    # Residual bookkeeping can leave negative values even on real
    # bidirectional edge pairs (net-direction cancellation) -- clip
    # to match nx's convention of nonnegative per-edge flow.
    filtered.data = np.clip(filtered.data, 0, None)
    filtered.eliminate_zeros()

    num_nodes = filtered.shape[0]
    flow_dict = {node_name: {} for node_name in _worker_idx_to_node.values()}

    indptr = filtered.indptr
    indices = filtered.indices
    data = filtered.data

    for u in range(num_nodes):
        for idx in range(indptr[u], indptr[u + 1]):
            v = indices[idx]
            flow_dict[_worker_idx_to_node[u]][_worker_idx_to_node[v]] = data[idx]

    return flow_dict


def calculate_flow_capacity(csr_masked, scale):
    """
    Calculates max flow capacity between source and sink.
    """
    res = maximum_flow(csr_masked, _worker_source_idx, _worker_sink_idx)
    flow_value = res.flow_value / scale
    flow_matrix = res.flow / scale
    flow_dict = scipy_to_flow_dict_active(flow_matrix, csr_masked)

    return flow_value, flow_dict


def scc_redundancy(csr_masked):
    # SCC redundancy (reuse same masked matrix — same trick as before)
    n_components, labels = connected_components(
        csr_masked, directed=True, connection="strong"
    )
    counts = np.bincount(labels)
    redundancy = int(np.sum(counts[counts > 1] ** 2))
    return redundancy


def evaluate(individual, scale, max_remove, constrained):
    """
    Fitness Function.
    Objectives (minimized):
    1. Flow Value (Max Flow)
    2. Number of Removed Nodes
    3. Co-occurrence in SCCs
    """
    csr_masked, removed_mask = mask_csr(individual)

    if constrained and removed_mask.sum() > max_remove:
        penalty_base = np.array([1e6, 1e6, 1e6])
        dist = float((removed_mask.sum() - max_remove) ** 2)
        return tuple(penalty_base + dist), {}

    # Max flow

    flow_value, _ = calculate_flow_capacity(csr_masked, scale)

    redundancy = scc_redundancy(csr_masked)

    cost = int(removed_mask.sum())

    return (float(flow_value), cost, redundancy), {}


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


# Save scaling for the edge capacities to work with int32:


def build_scaled_capacities(G, scale):
    """
    Returns (G_scaled, scale, sentinel_int) with capacities converted to
    safe int32-range integers. np.inf edges get an explicit finite
    sentinel, never a raw cast of infinity.
    """
    finite_caps = [
        d.get("capacity", 1.0)
        for _, _, d in G.edges(data=True)
        if np.isfinite(d.get("capacity", 1.0))
    ]
    max_finite = max(finite_caps)

    INT32_MAX = np.iinfo(np.int32).max

    # Safety check: does this scale keep the real (finite) capacities
    # comfortably inside int32 once scaled?
    max_scaled_finite = max_finite * scale
    if max_scaled_finite > INT32_MAX / 10:
        raise ValueError(
            f"scale={scale} pushes max finite capacity ({max_finite}) to "
            f"{max_scaled_finite}, too close to int32 limit ({INT32_MAX}). "
            f"Reduce scale."
        )

    # Sentinel: deliberately "big enough to never bottleneck", NOT a cast of inf.
    # 10x the largest real scaled capacity is comfortably non-bottlenecking
    # for any plausible network, while staying well under int32 max.
    sentinel = int(max_scaled_finite * 10)
    assert sentinel < INT32_MAX, "sentinel itself overflows int32 -- lower scale"

    G_scaled = G.copy()
    for u, v, d in G_scaled.edges(data=True):
        cap = d.get("capacity", 1.0)
        if not np.isfinite(cap):
            d["capacity"] = sentinel
        else:
            d["capacity"] = int(round(cap * scale))

    return G_scaled, sentinel


# --- class for loging on disk ---


class DiskLogger:
    """
    Streams per-generation records to a JSONL file instead of keeping
    them in memory. `visited_nodes` is stored as a diff (newly-visited
    nodes only) because the set only ever grows across generations —
    storing the full list every generation is O(N) per gen for no reason.
    """

    def __init__(self, log_path, flush_every=10):
        self.log_path = log_path
        self.flush_every = flush_every
        self._fh = open(log_path, "w", buffering=1)  # line-buffered
        self._n_since_flush = 0
        self._prev_visited = set()

    @staticmethod
    def _to_native(obj):
        """Recursively convert numpy scalars/arrays so json.dumps doesn't choke."""
        if isinstance(obj, dict):
            return {k: DiskLogger._to_native(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [DiskLogger._to_native(v) for v in obj]
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def log_generation(self, gen, record):
        record = dict(record)  # don't mutate caller's dict

        if "visited_nodes" in record:
            current = set(record["visited_nodes"])
            newly_visited = current - self._prev_visited
            self._prev_visited = current
            record["visited_nodes"] = sorted(newly_visited)  # diff only

        record = self._to_native(record)
        record["gen"] = gen

        self._fh.write(json.dumps(record) + "\n")
        self._n_since_flush += 1
        if self._n_since_flush >= self.flush_every:
            self._fh.flush()
            os.fsync(self._fh.fileno())
            self._n_since_flush = 0

    def close(self):
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._fh.close()


def load_logs(log_path):
    """
    Reconstructs the full {gen: record} dict from a JSONL log file,
    re-expanding visited_nodes from cumulative diffs. Use this after
    the run, for analysis/plotting — not during the run.
    """
    logs = {}
    cumulative = set()
    with open(log_path) as f:
        for line in f:
            rec = json.loads(line)
            gen = rec.pop("gen")
            if "visited_nodes" in rec:
                cumulative |= set(rec["visited_nodes"])
                rec["visited_nodes"] = sorted(cumulative)
            logs[gen] = rec
    return logs


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


class SmartCacheMap:
    """Orchestrator-level cache wrapper for evaluation dispatch.

    Checks a local dictionary before dispatching work to workers
    (multiprocessing pool or sequential), avoiding redundant computation
    for repeated individuals. Cache hits incur zero IPC overhead.
    """

    def __init__(self, evaluation_func, cache=None, pool=None):
        self.evaluation_func = evaluation_func
        self.cache = cache if cache is not None else {}
        self.pool = pool

    def __call__(self, func, iterable):
        """Dispatch evaluation with caching.

        Args:
            func: The evaluation function (registered in toolbox, ignored here
                  since self.evaluation_func is used directly).
            iterable: List of individuals to evaluate.

        Returns:
            List of (fitness_tuple, flow_dict) results.
        """
        results = [None] * len(iterable)
        to_compute_indices = []
        to_compute_values = []

        # Step A: Check local cache in main process (instant, zero IPC)
        for i, ind in enumerate(iterable):
            key = tuple(ind)
            if key in self.cache:
                results[i] = self.cache[key]
            else:
                to_compute_indices.append(i)
                to_compute_values.append(ind)

        # Step B: Dispatch only cache misses to workers
        if to_compute_values:
            if self.pool is not None:
                computed_results = self.pool.map(
                    self.evaluation_func, to_compute_values
                )
            else:
                computed_results = [self.evaluation_func(v) for v in to_compute_values]

            # Step C: Update cache and fill results
            for i, res in enumerate(computed_results):
                real_idx = to_compute_indices[i]
                val = to_compute_values[i]
                fit, _flow = res
                cached = (fit, {})
                self.cache[tuple(val)] = cached
                results[real_idx] = cached

        return results


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
        n_jobs=None,
        scale=1000,
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
        self.n_jobs = n_jobs

        if self.dynamic and not self.repair:
            print(
                "\nWARNING: Dynamic mutation rates will introduce a higher number of removed nodes. "
                "If repair=True is not set, it will lead to several individuals not respecting constraints.\n"
            )

        # FIX 2: Cleaner set subtraction for determining removable nodes
        self.removable_nodes = list(set(self.SG.nodes) - set(sources) - set(sinks))
        self.N = len(self.removable_nodes)

        # Build CSR one for all optimization
        self.node_order = list(self.SG.nodes)  # fixed order, defines CSR index space
        self.node_to_idx = {node: i for i, node in enumerate(self.node_order)}

        self.source_idx = self.node_to_idx["SuperSource"]
        self.sink_idx = self.node_to_idx["SuperSink"]

        # Precompute gene-index -> CSR-index mapping (removable_nodes order == gene order)
        self.removable_csr_indices = np.array(
            [self.node_to_idx[n] for n in self.removable_nodes], dtype=np.int64
        )

        self.scale = scale
        # Pre-scale capacity weights on the graph before conversion
        self.G_scaled, self.sentinel = build_scaled_capacities(
            G=self.SG, scale=self.scale
        )

        self.csr = nx.to_scipy_sparse_array(
            self.G_scaled,
            nodelist=self.node_order,
            weight="capacity",
            format="csr",
            dtype=np.int32,
        )

        # Set worker globals for baseline computation
        import sys as _sys

        _self_mod = _sys.modules[__name__]
        _self_mod._worker_csr = self.csr
        _self_mod._worker_removable_csr_indices = self.removable_csr_indices
        _self_mod._worker_source_idx = self.source_idx
        _self_mod._worker_sink_idx = self.sink_idx
        _self_mod._worker_idx_to_node = {v: k for k, v in self.node_to_idx.items()}
        # Calculate Baselines for Normalization
        (self.base_flow, _, self.base_redundancy), _ = evaluate(
            individual=[1] * self.N,
            scale=self.scale,
            max_remove=self.max_remove,
            constrained=False,
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
            creator.create(
                "Individual", array, typecode="B", fitness=creator.FitnessMin
            )

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

    def _get_n_jobs(self):
        """Return the number of parallel jobs to use for multiprocessing.

        If n_jobs is set (not None), use that value.
        Otherwise, fall back to multiprocessing.cpu_count().

        Raises:
            ValueError: If n_jobs is specified and is less than 1.
        """
        if self.n_jobs is not None:
            if self.n_jobs < 1:
                raise ValueError(f"n_jobs must be >= 1, got {self.n_jobs}")
            max_jobs = multiprocessing.cpu_count()
            if self.n_jobs > max_jobs:
                print(
                    f"Warning: n_jobs={self.n_jobs} exceeds CPU count ({max_jobs}), clamping to {max_jobs}"
                )
                return max_jobs
            return self.n_jobs
        return multiprocessing.cpu_count()

    def guess_parameters_gam(self):
        """
        guess fatures using GAM model
        """
        nf = NetworkFeatures(G=self.SG, b=self.max_remove)
        features = pd.DataFrame(nf.extract_features())
        gam = GAM()
        population, patience = gam.rank(df=features)
        return math.ceil(population), math.ceil(patience)

    def run(
        self,
        population_size=None,
        generations=1000,
        early_stop_gen=None,
        tol=1e-6,
        verbose=False,
        guess_parameters=True,
        log_path=None,
        log_flush_every=10,
        **kwargs,
    ):
        """
        Runs the Genetic Algorithm.
        """

        self.verbose = verbose

        if guess_parameters and all([population_size is None, early_stop_gen is None]):
            self.population_size, self.early_stop_gen = self.guess_parameters_gam()
            if self.verbose:
                print(
                    f"Guessed parameters:\nPopulation = {self.population_size}\nEarly_stop_gen = {self.early_stop_gen}"
                )
        elif guess_parameters and any(
            [population_size is not None, early_stop_gen is not None]
        ):
            raise ValueError(
                f"Set parameters guess: {guess_parameters} and specified population and/or early_stop_gen"
                f"Select if to guess or specify parameters"
                f"Population: {population_size}, early_stop_gen: {early_stop_gen}"
            )
        elif not guess_parameters and any(
            [population_size is None, early_stop_gen is None]
        ):
            raise ValueError(
                f"Need to guess or select parameters!"
                f"guess_parameters: {guess_parameters}"
                f"Population: {population_size}, early_stop_gen: {early_stop_gen}"
            )
        else:
            self.population_size = population_size
            self.early_stop_gen = early_stop_gen
        stagnation_trigger = self.early_stop_gen // 10
        rates = AdaptiveRates(
            base_cxpb=self.base_cxpb,
            base_mtpb=self.base_mtpb,
            base_indpb=self.base_indpb,
            stagnation_trigger=stagnation_trigger,
        )

        # setup logging file

        if log_path is None:
            log_path = f"ga_run_{int(time.time())}.jsonl"
        disk_logger = DiskLogger(log_path, flush_every=log_flush_every)
        self.log_path = log_path  # so callers can find it after run()
        summary_logs = {}

        pool = None
        local_cache = {}

        # Create evaluation function without shared_cache
        eval_func = functools.partial(
            evaluate,
            scale=self.scale,
            max_remove=self.max_remove,
            constrained=self.delta_penalty,
        )

        if self.use_multiprocessing:
            pool = multiprocessing.Pool(
                processes=self._get_n_jobs(),
                initializer=_init_worker,
                initargs=(
                    self.csr,
                    self.removable_csr_indices,
                    self.source_idx,
                    self.sink_idx,
                    self.node_to_idx,
                ),
                maxtasksperchild=200,
            )
            cache_aware_map = SmartCacheMap(
                evaluation_func=eval_func, cache=local_cache, pool=pool
            )
        else:
            cache_aware_map = SmartCacheMap(
                evaluation_func=eval_func, cache=local_cache, pool=None
            )

        self.toolbox.register("map", cache_aware_map)
        self.toolbox.register("evaluate", eval_func)

        try:
            population = self.toolbox.population(n=self.population_size)
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

                disk_logger.log_generation(gen, record)
                summary_logs[gen] = {
                    k: v for k, v in record.items() if k != "visited_nodes"
                }

                if self.verbose:
                    # FIX 7: Fixed KeyError. record["min"] doesn't exist anymore!
                    print(
                        f"{time.ctime()} - Gen {gen}: "
                        f"Flow: {record['Flow']}, Nodes: {record['Node']}, SCCs: {record['Redundancy']}. "
                        f"HV: {round(record['hv'], 3)}"
                    )

                if generations_without_improvement >= self.early_stop_gen:
                    if self.verbose:
                        print(
                            f"{time.ctime()} - Early stopping at generation {gen} (No HV improvement)."
                        )
                    break

        finally:
            disk_logger.close()
            if self.use_multiprocessing:
                if pool is not None:
                    pool.close()
                    pool.join()

        results, norm = select_candidates(
            population=population, removable_nodes=self.removable_nodes
        )

        return results, summary_logs, population, norm
