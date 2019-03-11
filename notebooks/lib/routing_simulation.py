import helper
import routing_algorithms

class Settings:
    def __init__(self, time_threshold = 10000, original_capacity = 1, original_cost = 1, long_link_cost = 1000,
                    rebuild_probability = 0.25, number_of_nodes = 32, number_of_source_destination_pairs = 50,
                    number_of_samples = 10):
        self.time_threshold = time_threshold
        self.original_capacity = original_capacity
        self.original_cost = original_cost
        self.long_link_cost = long_link_cost
        self.rebuild_probability = rebuild_probability
        self.number_of_nodes = number_of_nodes
        self.number_of_source_destination_pairs = number_of_source_destination_pairs
        self.number_of_samples = number_of_samples
        self.inf = float('inf')

# Execute the simulation for a distinct number of source and destination pairs multiple times
# graph: the graph in which we send the packets
# sd_pairs: number of source and destination pairs for which we are simulating for
# samples times: number of times we repeat the simulation
# algorithm: algorithm to be used to get the path and waiting time for a specific sd pair (default is Dijkstra)
def run_for_specific_source_destination_pair(graph_edges: list, sd_pairs: int, samples: int,
                                            algorithm = routing_algorithms.no_knowledge_init,
                                             algorithm_arguments = None, statistical_measure = helper.mean):
    results_for_source_destination = []
    for x in range(1, samples + 1):

        if algorithm_arguments is None:
            results: tuple = algorithm(graph_edges, sd_pairs, statistical_measure = statistical_measure)
        else:
            results: tuple = algorithm(graph_edges, sd_pairs, algorithm_arguments,
                                                                         statistical_measure = statistical_measure)

        results_for_source_destination.append(results)
    return helper.map_tuple_gen(statistical_measure, zip(*results_for_source_destination))


# Simulation for more than one run of the loop_for_specific_sd_pair
def run_for_specific_graph(graph_edges: list, sd_pairs: int, samples: int, algorithm=routing_algorithms.no_knowledge_init,
                           algorithm_arguments = None, statistical_measure = helper.mean):
    results_for_topology = []
    # Iterating through the remaining simulation rounds
    for x in range(1, sd_pairs + 1):
        results = run_for_specific_source_destination_pair(graph_edges, x, samples, algorithm, algorithm_arguments,
                                                           statistical_measure)

        # Summing up the values obtained in each round
        results_for_topology.append(results)

    # Returning the average of the result values
    return tuple(list(result) for result in zip(*results_for_topology))


def run_algorithm_for_graphs(graphs: tuple, number_of_sd_pairs, number_of_samples,
                            algorithm=routing_algorithms.no_knowledge_init,
                             algorithm_arguments = None , statistical_measure = helper.mean):
    return [run_for_specific_graph(graphs[graph_index], number_of_sd_pairs, number_of_samples, algorithm,
                                   algorithm_arguments, statistical_measure) for graph_index in range(len(graphs))]