import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../lib")

import helper
import routing_algorithms
import graph_edge_factory
import networkx as nx
import matplotlib.pyplot as plt


class Settings:
    def __init__(self, time_threshold = 10000, original_capacity = 1, original_cost = 1, long_link_cost = 1000,
                    rebuild_probability = 0.25, number_of_nodes = 32, number_of_source_destination_pairs = 50,
                    number_of_samples = 1000):
        self.time_threshold = time_threshold
        self.original_capacity = original_capacity
        self.original_cost = original_cost
        self.long_link_cost = long_link_cost
        self.rebuild_probability = rebuild_probability
        self.number_of_nodes = number_of_nodes
        self.number_of_source_destination_pairs = number_of_source_destination_pairs
        self.number_of_samples = number_of_samples
        self.inf = float('inf')


def extract_argument(key: str, argument_dictionary: dict, default_value):
    if key not in argument_dictionary.keys():
        extracted_argument = default_value
    else:
        extracted_argument = argument_dictionary[key]
    return extracted_argument


def draw_graph(graph_edges: list, dth: int, sample_number: int):

    G = nx.MultiGraph()
    G.add_edges_from(graph_edges)
    nx.draw_circular(G, with_labels=True)

    plt.savefig('/home/antal/Documents/eit/thesis/implementation/quantum_routing/notebooks/plots/random_power_law/'
                'graph_images/' + str(dth) + '_' + str(sample_number)+ '_' + '.png', bbox_inches='tight')
    plt.clf()
    return None


# Execute the simulation for a distinct number of source and destination pairs multiple times
# graph: the graph in which we send the packets
# sd_pairs: number of source and destination pairs for which we are simulating for
# samples times: number of times we repeat the simulation
# algorithm: algorithm to be used to get the path and waiting time for a specific sd pair (default is Dijkstra)
def run_for_specific_source_destination_pair(sd_pairs: int, samples: int, algorithm_arguments = None):
    results_for_source_destination = []
    link_length_dictionary = {}

    # Processing arguments

    # Extracting the algorithm
    algorithm = extract_argument('algorithm', algorithm_arguments, routing_algorithms.initial_knowledge_init)

    power_law = False
    # Extracting the model and the distance_threshold
    # if graph_edges are specified: then the topology is deterministically generated
    # Else: generate it based on the power-law distribution
    if 'graph_edges' in algorithm_arguments.keys():
        graph_edges = algorithm_arguments['graph_edges']
    else:
        power_law = True
        distance_threshold = extract_argument('distance_threshold', algorithm_arguments, 16)

    link_prediction = extract_argument('link_prediction', algorithm_arguments, False)
    exponential_scale = extract_argument('exponential_scale', algorithm_arguments, False)
    for x in range(1, samples + 1):

        if power_law:
            factory = graph_edge_factory.GraphEdgesFactory(distance_threshold=distance_threshold)
            graph_edges, link_lengths = factory.generate_random_power_law_graph_edges()
            # draw_graph(graph_edges, distance_threshold, x)

        if algorithm_arguments['algorithm'] == routing_algorithms.local_knowledge_algorithm and 'knowledge_radius' in \
                algorithm_arguments.keys():
            results: tuple = algorithm(graph_edges, sd_pairs, algorithm_arguments['knowledge_radius'],
                                       link_prediction=link_prediction, exponential_scale=exponential_scale)
        elif algorithm_arguments['algorithm'] == routing_algorithms.global_knowledge_init:
            results: tuple = algorithm(graph_edges, sd_pairs, exponential_scale=exponential_scale)
        else:
            results: tuple = algorithm(graph_edges, sd_pairs, link_prediction=link_prediction,
                                       exponential_scale=exponential_scale)

        if power_law:
            helper.add_dictionary_to_dictionary(link_length_dictionary, link_lengths)

        results_for_source_destination.append(results)
    return helper.map_tuple_gen(helper.mean, zip(*results_for_source_destination)), link_length_dictionary


# Simulation for more than one run of the loop_for_specific_sd_pair
def run_algorithm_for_graphs(sd_pairs: int, samples: int, arguments = None):
    results_for_topology = []
    link_length_dictionary = {}

    # Iterating through the remaining simulation rounds
    for x in range(1, sd_pairs + 1):
        results, link_lengths = run_for_specific_source_destination_pair(x, samples, arguments)

        # Summing up the values obtained in each round
        results_for_topology.append(results)
        helper.add_dictionary_to_dictionary(link_length_dictionary, link_lengths)

    # Returning the average of the result values
    return tuple(list(result) for result in zip(*results_for_topology)), link_length_dictionary
