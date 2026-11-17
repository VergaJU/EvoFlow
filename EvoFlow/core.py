"""Flow computation and the multi-objective genetic algorithm.

The graph is converted once to an ``int32`` CSR capacity matrix
(``scipy.sparse.csgraph.maximum_flow``); workers receive that matrix through
:func:`_init_worker` and evaluate individuals by zeroing the rows/columns of
removed nodes. Top-level functions must stay picklable for multiprocessing.

Objectives minimised by the GA are, in order:

1. maximum flow from source to sink,
2. number of removed nodes,
3. SCC redundancy (co-occurrence of removed nodes in strongly connected
   components).
"""

import functools
import json
import math
import multiprocessing
import os
import random
import time
from array import array

import networkx as nx
import numpy as np
import pandas as pd
from deap import base, creator, tools
from deap.benchmarks.tools import hypervolume
from scipy.sparse.csgraph import connected_components, maximum_flow
from scipy.spatial.distance import pdist

from EvoFlow.features_extraction import NetworkFeatures
from EvoFlow.model import GAM

_worker_csr = None
_worker_idx_to_node = None
_worker_removable_csr_indices = None
_worker_source_idx = None
_worker_sink_idx = None


def _init_worker(csr, removable_csr_indices, source_idx, sink_idx, node_to_idx):
    """Initialise module globals in a multiprocessing worker.

    The (potentially large) CSR matrix is shipped once per worker instead of
    with every task.

    Args:
        csr (scipy.sparse.csr_matrix): Scaled integer capacity matrix.
        removable_csr_indices (numpy.ndarray): CSR indices of removable nodes.
        source_idx (int): CSR index of the source.
        sink_idx (int): CSR index of the sink.
        node_to_idx (dict): Node name -> CSR index mapping.
    """
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
    """Extract the subgraph containing all paths between source and sink.

    Args:
        G (networkx.DiGraph): Graph with the super nodes.
        source (str): Source node name. Defaults to ``"SuperSource"``.
        sink (str): Sink node name. Defaults to ``"SuperSink"``.

    Returns:
        networkx.DiGraph: Copy of the subgraph induced by the nodes reachable
        from ``source`` that can also reach ``sink`` (plus both super nodes).
        An empty graph is returned when either super node is missing.
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
    """Generate a random individual as a binary mask.

    ``1`` means keep the node and ``0`` means remove it. The number of removed
    nodes is drawn uniformly from ``1..max_remove``.

    Args:
        icls (type): Individual class registered with DEAP (``creator.Individual``).
        removable_nodes (list): Nodes the mask indexes.
        max_remove (int): Maximum number of nodes to remove. Defaults to ``3``.

    Returns:
        deap.creator.Individual: Binary individual.
    """
    ind = [1] * len(removable_nodes)
    n_remove = min(random.randint(1, max_remove), len(removable_nodes))

    to_remove = random.sample(range(len(removable_nodes)), n_remove)
    for i in to_remove:
        ind[i] = 0
    return icls(array("B", ind))


def repair_individual(ind, min_remove=2, max_remove=10):
    """Repair an individual so its number of removals is within bounds.

    Random ``0 -> 1`` or ``1 -> 0`` flips are applied until the count of
    removed nodes lies in ``[min_remove, max_remove]``. The fitness values are
    invalidated when the individual is modified.

    Args:
        ind (deap.creator.Individual): Individual to repair in place.
        min_remove (int): Minimum number of removed nodes. Defaults to ``2``.
        max_remove (int): Maximum number of removed nodes. Defaults to ``10``.

    Returns:
        deap.creator.Individual: The repaired individual (same object).
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
    """Flip each bit of an individual with probability ``indpb``.

    Args:
        individual (deap.creator.Individual): Individual to mutate in place.
        indpb (float): Per-bit flip probability. Defaults to ``0.05``.

    Returns:
        tuple: One-element tuple containing the mutated individual, as
        required by DEAP.
    """
    tools.mutFlipBit(individual, indpb)
    # cleanup_individual(individual, min_remove, max_remove)
    return (individual,)


# --- Util get nodes --


def get_nodes(removable_nodes, individual):
    """Map a binary individual back to the list of removed node names.

    Args:
        removable_nodes (list): Nodes indexed by the individual.
        individual (Sequence[int]): Binary mask where ``0`` means removed.

    Returns:
        list: Names of the removed nodes.
    """
    return [node for i, node in enumerate(removable_nodes) if individual[i] == 0]


# --- Evaluation Function (Must be top-level for multiprocessing) ---


def mask_csr(individual):
    """Zero out the rows and columns of removed nodes in the worker CSR matrix.

    Args:
        individual (Sequence[int]): Binary mask where ``0`` means removed.

    Returns:
        tuple: ``(csr_masked, removed_mask)`` where ``csr_masked`` is a copy of
        the worker capacity matrix with removed nodes disconnected and
        ``removed_mask`` is the boolean removal mask.
    """
    removed_mask = ~np.array(individual, dtype=bool)
    removed_csr_idx = _worker_removable_csr_indices[removed_mask]

    csr_masked = _worker_csr.copy()
    csr_masked[removed_csr_idx, :] = 0
    csr_masked[:, removed_csr_idx] = 0
    csr_masked.eliminate_zeros()
    return csr_masked, removed_mask


def scipy_to_flow_dict_active(flow_matrix, csr_masked):
    """
    Convert ``MaximumFlowResult.flow`` to a dict of active flows on real edges.

    The result contains only positive flows on edges that exist in the
    original capacity matrix, matching ``networkx.maximum_flow`` semantics.

    ``flow_matrix`` includes scipy's internally added reverse/residual edges
    (with zero or negative bookkeeping values); these are filtered out using
    the original capacity structure, not the flow matrix's own sparsity.

    Args:
        flow_matrix (scipy.sparse.spmatrix): Flow values returned by
            ``scipy.sparse.csgraph.maximum_flow`` (already divided by ``scale``).
        csr_masked (scipy.sparse.csr_matrix): Masked capacity matrix.

    Returns:
        dict: ``{u: {v: flow}}`` mapping for every real edge with positive flow.
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
    """Compute the maximum flow between the worker source and sink.

    Args:
        csr_masked (scipy.sparse.csr_matrix): Masked integer capacity matrix.
        scale (int): Integer scaling factor used when the matrix was built.

    Returns:
        tuple[float, dict]: ``(flow_value, flow_dict)`` where ``flow_value`` is
        rescaled back to the original capacity units and ``flow_dict`` is the
        per-edge flow mapping.
    """
    res = maximum_flow(csr_masked, _worker_source_idx, _worker_sink_idx)
    flow_value = res.flow_value / scale
    flow_matrix = res.flow / scale
    flow_dict = scipy_to_flow_dict_active(flow_matrix, csr_masked)

    return flow_value, flow_dict


def scc_redundancy(csr_masked):
    """Measure redundancy as the sum of squared sizes of multi-node SCCs.

    Args:
        csr_masked (scipy.sparse.csr_matrix): Masked capacity matrix.

    Returns:
        int: ``sum(size ** 2)`` over strongly connected components with more
        than one node (``0`` when no redundant components remain).
    """
    # SCC redundancy (reuse same masked matrix — same trick as before)
    n_components, labels = connected_components(
        csr_masked, directed=True, connection="strong"
    )
    counts = np.bincount(labels)
    redundancy = int(np.sum(counts[counts > 1] ** 2))
    return redundancy


def evaluate(individual, scale, max_remove, constrained):
    """Fitness function for the GA (must be top-level for multiprocessing).

    Objectives, all minimised:

    1. maximum flow value,
    2. number of removed nodes,
    3. SCC co-occurrence (redundancy).

    Args:
        individual (Sequence[int]): Binary removal mask.
        scale (int): Integer scaling factor of the capacity matrix.
        max_remove (int): Maximum allowed number of removals.
        constrained (bool): When true, individuals above ``max_remove`` receive
            a large squared penalty instead of being evaluated.

    Returns:
        tuple: ``(fitness_tuple, {})`` as expected by DEAP's ``map``.
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
    """Mean pairwise Hamming distance of the population.

    Args:
        population (list): DEAP individuals (binary masks).

    Returns:
        float: Mean genotypic diversity, or ``0.0`` for populations with fewer
        than two individuals.
    """
    if len(population) < 2:
        return 0.0
    pop_array = np.array(population, dtype=int)

    distances = pdist(pop_array, metric="hamming")
    return np.mean(distances)


# --- Get visited nodes per generation ---


def collapse_pop(population):
    """Collapse a population to the set of nodes removed by *every* individual.

    Args:
        population (list): DEAP individuals (binary masks).

    Returns:
        numpy.ndarray: Binary mask that is ``1`` only where all individuals
        kept the node.
    """
    pop_array = np.array(population, dtype=int)
    # Collapse: Result is 1 only if ALL are 1. If any are 0, result is 0.
    collapsed = np.all(pop_array, axis=0).astype(int)
    return collapsed


# --- Select candidates ---


def select_candidates(population, removable_nodes):
    """Pick the four reported solutions from the Pareto front.

    The front is normalised per objective and the balanced solution is the one
    closest to the origin; the remaining three minimise flow, cost and
    redundancy individually.

    Args:
        population (list): DEAP individuals with valid fitness values.
        removable_nodes (list): Nodes indexed by the individuals.

    Returns:
        tuple[dict, numpy.ndarray]: ``(results, norm)`` where ``results`` maps
        ``Best_balanced``, ``Best_Flow``, ``Best_Cost`` and ``Best_Redundancy``
        to dicts with ``fitness``, ``num_removed`` and ``removed_nodes``, and
        ``norm`` is the normalized front used for the TSV export.
    """
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
    """Select the most crowded individuals of the first Pareto front.

    Lowest crowding distance means least diverse; these individuals are the
    ones mutated when the run stagnates.

    Args:
        population (list): DEAP individuals with crowding distances assigned.
        n (int): Number of individuals to return.

    Returns:
        list: The ``n`` individuals with the smallest crowding distance.
    """
    front = tools.sortNondominated(population, len(population), first_front_only=True)[
        0
    ]
    # Sort by crowding distance ascending (most crowded first)
    sorted_front = sorted(front, key=lambda ind: ind.fitness.crowding_dist)
    return sorted_front[:n]  # Return the most crowded ones to mutate


# Save scaling for the edge capacities to work with int32:


def build_scaled_capacities(G, scale):
    """Convert capacities to safe ``int32`` integers for scipy's max-flow.

    Infinite capacities (super-node edges) get an explicit finite sentinel
    value rather than a raw cast of infinity.

    Args:
        G (networkx.DiGraph): Graph with a ``capacity`` edge attribute.
        scale (int): Multiplicative factor applied before rounding.

    Returns:
        tuple[networkx.DiGraph, int]: ``(G_scaled, sentinel)`` where
        ``G_scaled`` is a copy with integer capacities and ``sentinel`` is the
        value assigned to formerly infinite capacities.

    Raises:
        ValueError: If ``scale`` pushes the largest finite capacity too close
            to the ``int32`` limit.
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
    Streams per-generation records to a JSONL file instead of keeping them in memory.

    ``visited_nodes`` is stored as a diff (newly visited nodes only) because the
    set only ever grows across generations — storing the full list every
    generation would be O(N) per generation for no benefit.

    Attributes:
        log_path (str | pathlib.Path): Destination JSONL file.
    """

    def __init__(self, log_path, flush_every=10):
        """Open the log file for writing.

        Args:
            log_path (str | pathlib.Path): Destination JSONL file (truncated).
            flush_every (int): Number of records between explicit flushes.
                Defaults to ``10``.
        """
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
        """Append one generation record to the log file.

        Args:
            gen (int): Generation number.
            record (dict): Metrics for the generation. ``visited_nodes`` is
                rewritten as a diff against the previous record.
        """
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
        """Flush and close the underlying log file."""
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._fh.close()


def load_logs(log_path):
    """
    Reconstruct the full ``{gen: record}`` dict from a JSONL log file.

    ``visited_nodes`` is re-expanded from the cumulative diffs written by
    :class:`DiskLogger`. Use this after the run for analysis/plotting, not
    during it.

    Args:
        log_path (str | pathlib.Path): JSONL file written during a run.

    Returns:
        dict: Mapping ``generation -> record``.
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
    """Mutation/crossover rates that decay and periodically reheat.

    Rates decay geometrically toward their minima while the run is improving.
    Once a minimum is reached and the run has stagnated for
    ``stagnation_trigger`` generations, the rates are heated back toward their
    base values, with each reheating spike attenuated by
    ``reheat_attenuation``.
    """

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
        """Initialise the rates.

        Args:
            base_cxpb (float): Starting (and peak) crossover probability.
            base_mtpb (float): Starting (and peak) mutation probability.
            base_indpb (float): Starting per-bit mutation probability.
            min_cxpb (float): Lower bound for crossover. Defaults to ``0.5``.
            min_mtpb (float): Lower bound for mutation. Defaults to ``0.1``.
            min_indpb (float, optional): Lower bound for per-bit mutation.
                Defaults to half of ``base_indpb``.
            max_cxpb (float): Upper bound for crossover. Defaults to ``0.95``.
            max_mtpb (float): Upper bound for mutation. Defaults to ``0.8``.
            max_indpb (float, optional): Upper bound for per-bit mutation.
                Defaults to ``min(5 * base_indpb, 0.5)``.
            decay (float): Per-generation multiplicative decay. Defaults to
                ``0.997``.
            stagnation_trigger (int): Generations without improvement before
                reheating. Defaults to ``3``.
            reheat_attenuation (float): Factor applied to the spike height at
                each reheating. Defaults to ``0.9``.
        """
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
        """Advance the rates by one generation.

        Args:
            generations_without_improvement (int): Number of consecutive
                generations without fitness improvement.
        """
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
        Raise the rates toward their base values with an attenuated spike.

        Every reheating produces a lower peak than the previous one, keeping the
        rates above their minima while gradually reducing exploration.
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
    """Stand-in individual exposing only a ``fitness.values`` attribute.

    Used to reuse the Pareto-normalisation code on plain fitness vectors.
    """

    def __init__(self, values):
        """Wrap a fitness tuple.

        Args:
            values (tuple): Fitness values to expose as ``self.fitness.values``.
        """
        # We create a fake fitness object with a values attribute
        self.fitness = type("Fitness", (), {"values": values})


class SmartCacheMap:
    """Orchestrator-level cache wrapper for evaluation dispatch.

    Checks a local dictionary before dispatching work to workers
    (multiprocessing pool or sequential), avoiding redundant computation
    for repeated individuals. Cache hits incur zero IPC overhead.
    """

    def __init__(self, evaluation_func, cache=None, pool=None):
        """Wrap an evaluation function with a main-process cache.

        Args:
            evaluation_func (callable): Function mapping an individual to
                ``(fitness_tuple, flow_dict)``.
            cache (dict, optional): Cache shared across calls. A new dict is
                created when omitted.
            pool (multiprocessing.pool.Pool, optional): Worker pool used for
                cache misses. When ``None``, evaluation runs sequentially.
        """
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
    """Multi-objective genetic algorithm over node-removal sets.

    The active subgraph (all paths between ``SuperSource`` and ``SuperSink``) is
    converted once to an integer CSR capacity matrix. Individuals are binary
    masks over the removable nodes and are evaluated with
    :func:`evaluate`; results are cached by :class:`SmartCacheMap`.

    Attributes:
        G (networkx.DiGraph): Input graph.
        SG (networkx.DiGraph): Active subgraph the GA actually optimises.
        sources (list): Source nodes (protected from removal).
        sinks (list): Sink nodes (protected from removal).
        removable_nodes (list): Nodes the GA may remove.
        N (int): Number of removable nodes.
        node_order (list): Node order defining the CSR index space.
        csr (scipy.sparse.csr_matrix): Scaled integer capacity matrix.
        base_flow (float): Maximum flow before any removal, used for
            normalisation.
        population_size (int): Population size used by the last
            :meth:`run` call.
        early_stop_gen (int): Early-stopping patience used by the last
            :meth:`run` call.
    """

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
        """Prepare the graph, DEAP toolbox and evaluation machinery.

        Args:
            G (networkx.DiGraph): Graph prepared with
                :func:`EvoFlow.preparer.prepare_flow_topology`.
            sources (list): Source nodes; never removed.
            sinks (list): Sink nodes; never removed.
            max_remove (int): Maximum removals per individual. Defaults to
                ``10``.
            min_remove (int): Minimum removals per individual. Defaults to
                ``1``.
            delta_penalty (bool): Apply the over-budget penalty during
                evaluation. Defaults to ``False``.
            repair (bool): Repair individuals to respect the removal bounds.
                Defaults to ``False``.
            norm_hypervolume (bool): Normalise the hypervolume reference.
                Defaults to ``False``.
            dynamic (bool): Use adaptive crossover/mutation rates. Requires
                ``repair=True`` to keep individuals valid. Defaults to
                ``False``.
            base_cxpb (float): Initial crossover probability. Defaults to
                ``0.75``.
            base_mtpb (float): Initial mutation probability. Defaults to
                ``0.9``.
            base_indpb (int): Per-bit mutation percentage used to derive the
                per-bit probability. Defaults to ``5``.
            use_multiprocessing (bool): Evaluate in a process pool. Defaults to
                ``True``.
            verbose (bool): Print progress messages. Defaults to ``False``.
            n_jobs (int, optional): Worker count. ``None`` uses all CPUs.
            scale (int): Integer capacity scaling. Defaults to ``1000``.
            **kwargs: Accepted for backwards compatibility and ignored.
        """
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
        """Predict GA hyper-parameters with the bundled GAM model.

        Extracts the four network features with
        :class:`~EvoFlow.features_extraction.NetworkFeatures` and ranks
        population-size/patience candidates with :class:`~EvoFlow.model.GAM`.

        Returns:
            tuple[int, int]: ``(population_size, early_stop_gen)`` rounded up to
            the next integer.
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
        """Run the genetic algorithm.

        Args:
            population_size (int, optional): GA population size. Must be given
                together with ``early_stop_gen``; omit both to let the GAM guess
                them.
            generations (int): Maximum number of generations. Defaults to
                ``1000``.
            early_stop_gen (int, optional): Generations without improvement
                before stopping.
            tol (float): Improvement tolerance for stagnation. Defaults to
                ``1e-6``.
            verbose (bool): Print progress messages. Defaults to ``False``.
            guess_parameters (bool): Estimate the GA parameters with the GAM
                when they are omitted. Defaults to ``True``.
            log_path (str, optional): Destination JSONL log file. Defaults to
                ``ga_run_<epoch>.jsonl`` in the current working directory.
            log_flush_every (int): Records between explicit flushes. Defaults to
                ``10``.
            **kwargs: Accepted for backwards compatibility and ignored.

        Returns:
            tuple: ``(results, summary_logs, population, norm)`` where
            ``results`` maps the four reported solutions to their fitness and
            removed nodes, ``summary_logs`` holds per-generation metrics,
            ``population`` is the final DEAP population and ``norm`` is the
            normalized Pareto front.

        Raises:
            ValueError: If ``guess_parameters`` is set together with only one of
                ``population_size``/``early_stop_gen``, or if guessing is
                disabled while either value is missing.
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
