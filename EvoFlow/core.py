import random
import time
import networkx as nx
import multiprocessing
from deap import base, creator, tools
from deap.benchmarks.tools import hypervolume
import functools
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
            redundancy += len(scc)**2
    return redundancy

# def receptor_targets_coocurrence(sccs, receptors, target_genes):
#     """
#     Calculates co-occurrence of receptors and target genes in SCCs.
#     """
#     coocurrence = 0
#     receptors_set = set(receptors)
#     targets_set = set(target_genes)
    
#     for scc in sccs:
#         nodes = list(scc)
#         # Find receptors and target genes in the SCC
#         receptors_sccs = [node for node in nodes if node in receptors_set]
#         targets_sccs = [node for node in nodes if node in targets_set]
        
#         # Calculate co-occurrence
#         coocurrence += len(receptors_sccs) * len(targets_sccs)
#     return coocurrence

def calculate_flow_capacity(G, ssource, ssink):
    """
    Calculates max flow capacity between source and sink.
    """
    try:
        flow_value = nx.maximum_flow_value(G, ssource, ssink, capacity='capacity')
    except Exception as e:
        # Fallback or error handling if graph structure is invalid
        print(f"Error calculating flow: {e}")
        flow_value = 0
    return flow_value

def extract_active_subgraph(G, source='SuperSource', sink='SuperSink'):
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

# --- Evaluation Function (Must be top-level for multiprocessing) ---

def evaluate(individual, G, removable_nodes, ssource, ssink):
    """
    Fitness Function.
    Objectives (minimized):
    1. Flow Value (Max Flow)
    2. Number of Removed Nodes
    3. Co-occurrence in SCCs
    """
    # Identify nodes to remove
    removed_nodes_names = [node for i, node in enumerate(removable_nodes) if individual[i] == 0]
    
    # Create temp graph
    G_copy = G.copy()
    G_copy.remove_nodes_from(removed_nodes_names)
    
    # 1. Flow Capacity
    flow_value = calculate_flow_capacity(G_copy, ssource, ssink)
    
    # 2. SCC Co-occurrence
    sccs = find_scc(G_copy)
    redundancy = scc_redundancy(sccs=sccs)
    # 3. Cost (Number of nodes removed)
    cost = len(removed_nodes_names)

    return flow_value, cost, redundancy

# --- Constraint Helpers ---

def check_feasible(individual, max_remove):
    """Returns True if the individual removes no more than max_remove nodes."""
    return individual.count(0) <= max_remove

def penalty_distance(individual, max_remove):
    """Calculates how far the individual is from being feasible."""
    return float((individual.count(0) - max_remove) ** 2)

def constrained_evaluate(individual, G, removable_nodes, ssource, ssink, max_remove):
    """
    A manual wrapper that mimics DeltaPenalty but is picklable.
    """
    if not check_feasible(individual, max_remove):
        # Penalty values (must be a tuple for multi-objective)
        penalty_base = (1e6, 1e6, 1e6)
        dist = penalty_distance(individual, max_remove)
        return tuple(p + dist for p in penalty_base)
    
    # Otherwise, return actual fitness
    return evaluate(individual, G, removable_nodes, ssource, ssink)

# --- MockIndividual to normalize front ---
class MockIndividual:
    def __init__(self, values):
        # We create a fake fitness object with a values attribute
        self.fitness = type('Fitness', (), {'values': values})
# --- GA Engine Class ---

class GAOptimizer:
    def __init__(self, 
                 G, 
                 sources, 
                 sinks, 
                 max_remove = 10, 
                 min_remove = 1,
                 constrained=False,
                 repair = False,
                 norm_hypervolume = False):
        self.G = G
        self.sources = sources
        self.sinks = sinks
        self.min_remove = min_remove
        self.max_remove=max_remove
        self.constrained = constrained
        self.repair = repair
        self.norm_hypervolume = norm_hypervolume
        # Determine removable nodes (All nodes except Sources and Sinks)
        self.removable_nodes = list(set(set(G.nodes) - set(sources)) - set(sinks))
        
        # Determine max flow and SCC:
        # Calculate Baselines for Normalization
        # Use the original graph (0 nodes removed) as the "Max" reference
        self.base_flow, _, self.base_redundancy = evaluate([1] * len(self.removable_nodes), # Individual where all nodes are kept (all 1s)
                                                           self.G,
                                                           self.removable_nodes,
                                                           'SuperSource',
                                                           'SuperSink'
                                                           )
        
        # Handle cases where base values might be 0 to avoid DivisionByZero
        self.base_flow = max(self.base_flow, 1.0)
        self.base_redundancy = max(self.base_redundancy, 1.0)
        self.base_cost = max(max_remove, 1.0)
        self.N = len(self.removable_nodes)
        # DEAP Setup
        # Use a unique name for creator classes to avoid conflicts if re-instantiated
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)
            
        self.toolbox = base.Toolbox()
        
        # Registration
        self.toolbox.register("individual", generate_individual, 
                              icls=creator.Individual, 
                              removable_nodes=self.removable_nodes, 
                              max_remove=self.max_remove)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        

        if self.constrained:
            self.toolbox.register("evaluate", constrained_evaluate, 
                                G=self.G, 
                                removable_nodes=self.removable_nodes, 
                                ssource='SuperSource', 
                                ssink='SuperSink',
                                max_remove=self.max_remove)
        else:
            self.toolbox.register("evaluate", evaluate, 
                                G=self.G, 
                                removable_nodes=self.removable_nodes, 
                                ssource='SuperSource', 
                                ssink='SuperSink')

        self.toolbox.register("mate", tools.cxUniform, indpb=0.5)
        self.toolbox.register("mutate", mutate, indpb=1.0/self.N)
        
        self.toolbox.register("repair", repair_individual, min_remove=self.min_remove, max_remove=self.max_remove)

        # change selection to NSGA for real multi-objective
        self.toolbox.register("select", tools.selNSGA2)
        # self.toolbox.register("select", tools.selTournament, tournsize=3)


    def normalize_fitness(self, fitness_tuple):
            """
            Scales fitness values to [0, 1] range.
            Original: (Flow, Cost, Redundancy)
            """
            f0 = fitness_tuple[0] / self.base_flow
            f1 = fitness_tuple[1] / self.base_cost
            f2 = fitness_tuple[2] / self.base_redundancy
            
            norm_fitness_tuple = (min(f0, 1.0), min(f1, 1.0), min(f2, 1.0))
            # Clip values to [0, 1] just in case logic results in slightly higher values
            return norm_fitness_tuple

    def run(self, 
            population_size=300, 
            generations=100, 
            early_stop_gen=10, 
            tol=1e-6, 
            use_multiprocessing=True,
            verbose=False):
        """
        Runs the Genetic Algorithm.
        """
        if use_multiprocessing:
            pool = multiprocessing.Pool()
            self.toolbox.register("map", pool.map)
        
        try:
            population = self.toolbox.population(n=population_size)
            print(f"{time.ctime()} - Evaluating fitness for initial population...")
            
            # Evaluate initial population
            invalid_ind = [ind for ind in population if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
                
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            # stats.register("avg", lambda x: sum([i[0] for i in x]) / len(x))
            stats.register("min", min)
            # stats.register("max", max)
            ref_point = (1.1, 1.1, 1.1) 
            best_hv = 0.0

            generations_without_improvement = 0
            logs = {}
            
            print(f"{time.ctime()} - Starting evolution for {generations} generations")
            
            for gen in range(generations):
                # Selection
                offspring = self.toolbox.select(population, len(population))
                offspring = list(map(self.toolbox.clone, offspring))
                
                # Crossover
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < 0.9:
                        self.toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values

                # Mutation
                for mutant in offspring:
                    if random.random() < 0.3:
                        self.toolbox.mutate(mutant)
                        del mutant.fitness.values
    
                # is repair is set as True, then repair not viable individuals
                if self.repair:
                    for mutant in offspring:
                        self.toolbox.repair(mutant)

                # Evaluation
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit
                    
                # Replacement
                # Include parents for elitism
                population[:] = self.toolbox.select(population + offspring, len(population))
                
                # Stats
                record = stats.compile(population)
                if verbose:
                    print(f"Gen {gen}: {record}")
                logs[gen] = record
                
                # Early Stopping
                feasible_pop = [ind for ind in population if ind.fitness.values[0] < 1e5]
                
                if feasible_pop:
                    front = tools.sortNondominated(feasible_pop, len(feasible_pop), first_front_only=True)[0].copy()

                    if self.norm_hypervolume:
                        normalized_mock_front = []
                        for ind in front:
                            # Create a dummy object
                            mock_ind = type('MockIndividual', (), {})()
                            
                            # Instantiate a REAL Fitness object of the same class you used for the GA
                            # This will have the 'weights' and 'wvalues' logic built-in
                            mock_ind.fitness = creator.FitnessMin()
                            
                            # Normalize the values
                            norm_values = self.normalize_fitness(ind.fitness.values)
                            
                            # Setting .values automatically triggers DEAP to calculate .wvalues
                            mock_ind.fitness.values = norm_values
                            
                            normalized_mock_front.append(mock_ind)
        
                        current_hv = hypervolume(normalized_mock_front, ref_point)
                    else:
                        current_hv = hypervolume(front)

                    if current_hv > best_hv + tol:
                        best_hv = current_hv
                        generations_without_improvement = 0
                    else:
                        generations_without_improvement += 1
                else:
                    # If no feasible individuals yet, we don't increment/decrement early stopping
                    current_hv = 0.0

                # --- STATS AND LOGGING ---
                record = stats.compile(population)
                record['hv'] = current_hv  # Track HV in logs
                logs[gen] = record
                
                if verbose:
                    print(f"Gen {gen}: HV={current_hv:.4f}, {record}")

                if generations_without_improvement >= early_stop_gen:
                    print(f"Early stopping at generation {gen} (No HV improvement).")
                    break
                    
        finally:
            if use_multiprocessing:
                pool.close()
                pool.join()
                
        best_ind = tools.selBest(population, 1)[0]
        
        # Convert best individual (binary mask) back to node names
        best_nodes_removed = [node for i, node in enumerate(self.removable_nodes) if best_ind[i] == 0]
        
        return best_nodes_removed, logs, population
