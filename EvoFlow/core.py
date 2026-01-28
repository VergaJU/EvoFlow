import random
import time
import networkx as nx
import multiprocessing
from deap import base, creator, tools

# --- Helper Functions (Graph Algorithms) ---

def find_scc(G):
    """
    Find strongly connected components in a directed graph.
    Returns a list of sets, where each set contains nodes in a component (len > 1).
    """
    sccs = [i for i in nx.strongly_connected_components(G) if len(i) > 1]
    return sccs

def receptor_targets_coocurrence(sccs, receptors, target_genes):
    """
    Calculates co-occurrence of receptors and target genes in SCCs.
    """
    coocurrence = 0
    receptors_set = set(receptors)
    targets_set = set(target_genes)
    
    for scc in sccs:
        nodes = list(scc)
        # Find receptors and target genes in the SCC
        receptors_sccs = [node for node in nodes if node in receptors_set]
        targets_sccs = [node for node in nodes if node in targets_set]
        
        # Calculate co-occurrence
        coocurrence += len(receptors_sccs) * len(targets_sccs)
    return coocurrence

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
    n_remove = random.randint(2, max_remove)
    # Ensure we don't try to remove more than exist
    n_remove = min(n_remove, len(removable_nodes))
    
    to_remove = random.sample(range(len(removable_nodes)), n_remove)
    for i in to_remove:
        ind[i] = 0
    return icls(ind)

def cleanup_individual(ind, min_removed=2, max_removed=5):
    """
    Repairs an individual to stay within deletion bounds.
    """
    current_removed = sum(1 for gene in ind if gene == 0)
    
    # Too many deletions
    if current_removed > max_removed:
        zero_indices = [i for i, g in enumerate(ind) if g == 0]
        to_flip = random.sample(zero_indices, current_removed - max_removed)
        for i in to_flip:
            ind[i] = 1

    # Too few deletions
    elif current_removed < min_removed:
        one_indices = [i for i, g in enumerate(ind) if g == 1]
        # Ensure we have enough ones to flip
        if len(one_indices) >= (min_removed - current_removed):
            to_flip = random.sample(one_indices, min_removed - current_removed)
            for i in to_flip:
                ind[i] = 0
    
    return ind

def mutate_shuffle(individual, indpb=0.05, min_removed=2, max_removed=10):
    """
    Wrapper for shuffle mutation that includes cleanup.
    """
    tools.mutShuffleIndexes(individual, indpb)
    cleanup_individual(individual, min_removed, max_removed)
    return (individual,)

# --- Evaluation Function (Must be top-level for multiprocessing) ---

def evaluate(individual, G, removable_nodes, ssource, ssink, receptors, target_genes):
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
    cooccurrence = receptor_targets_coocurrence(sccs, receptors, target_genes)
    
    # 3. Cost (Number of nodes removed)
    cost = len(removed_nodes_names)

    return flow_value, cost, cooccurrence

# --- GA Engine Class ---

class GAOptimizer:
    def __init__(self, G, sources, sinks, receptors, target_genes):
        self.G = G
        self.sources = sources
        self.sinks = sinks
        self.receptors = receptors
        self.target_genes = target_genes
        
        # Determine removable nodes (All nodes except Sources and Sinks)
        self.removable_nodes = list(set(set(G.nodes) - set(sources)) - set(sinks))
        
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
                              max_remove=10)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        self.toolbox.register("evaluate", evaluate, 
                              G=self.G, 
                              removable_nodes=self.removable_nodes, 
                              ssource='SuperSource', 
                              ssink='SuperSink',
                              receptors=self.receptors,
                              target_genes=self.target_genes)
                              
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", mutate_shuffle, indpb=0.05, min_removed=2, max_removed=10)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def run(self, population_size=300, generations=100, early_stop_gen=10, tol=1e-6, use_multiprocessing=True):
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
            stats.register("avg", lambda x: sum([i[0] for i in x]) / len(x))
            stats.register("min", min)
            stats.register("max", max)
            
            best_fitness = None
            generations_without_improvement = 0
            logs = {}
            
            print(f"{time.ctime()} - Starting evolution for {generations} generations")
            
            for gen in range(generations):
                # Selection
                offspring = self.toolbox.select(population, len(population))
                offspring = list(map(self.toolbox.clone, offspring))
                
                # Crossover
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < 0.5:
                        self.toolbox.mate(child1, child2)
                        cleanup_individual(child1, min_removed=2, max_removed=10)
                        cleanup_individual(child2, min_removed=2, max_removed=10)
                        del child1.fitness.values
                        del child2.fitness.values

                # Mutation
                for mutant in offspring:
                    if random.random() < 0.2:
                        self.toolbox.mutate(mutant)
                        del mutant.fitness.values
                        
                # Evaluation
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit
                    
                # Replacement
                population[:] = offspring
                
                # Stats
                record = stats.compile(population)
                print(f"Gen {gen}: {record}")
                logs[gen] = record
                
                # Early Stopping
                current_best_fitness = min([ind.fitness.values[0] for ind in population])
                if best_fitness is None or current_best_fitness < best_fitness - tol:
                    best_fitness = current_best_fitness
                    generations_without_improvement = 0
                else:
                    generations_without_improvement += 1
                    
                if generations_without_improvement >= early_stop_gen:
                    print(f"Early stopping at generation {gen}.")
                    break
                    
        finally:
            if use_multiprocessing:
                pool.close()
                pool.join()
                
        best_ind = tools.selBest(population, 1)[0]
        
        # Convert best individual (binary mask) back to node names
        best_nodes_removed = [node for i, node in enumerate(self.removable_nodes) if best_ind[i] == 0]
        
        return best_nodes_removed, logs
