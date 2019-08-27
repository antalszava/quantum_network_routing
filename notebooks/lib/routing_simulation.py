import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../lib")

import helper
import routing_algorithms
import graph_edge_factory
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss

class Settings:
    """
    Generic class for simulation settings.

    Attributes
    ----------
    time_threshold: int
        The time threshold specified for the decoherence time.

    original_capacity: int
        Original capacity of links.

    original_cost: int
        Original cost of links.

    long_link_cost: int
        Cost of long links.

    rebuild_probability: float
        Probability of rebuilding a link in a given time window.

    number_of_nodes: int
        The number of nodes of the graph in the simulation.

    number_of_source_destination_pairs: int
        The number of nodes of the graph in the simulation.

    number_of_samples: int
        Number of times the simulation is repeated.

    inf: float
        The value of infinity used in finding the shortest path.
    """
    def __init__(self, time_threshold = 1000, original_capacity = 1, original_cost = 1, long_link_cost = 1000,
                    rebuild_probability = 0.25, number_of_nodes = 32, number_of_source_destination_pairs = 50,
                    number_of_samples = 1000):
        """
        Generic class for simulation settings.

        Parameters
        ----------
        time_threshold: int
            The time threshold specified for the decoherence time.

        original_capacity: int
            Original capacity of links.

        original_cost: int
            Original cost of links.

        long_link_cost: int
            Cost of long links.

        rebuild_probability: float
            Probability of rebuilding a link in a given time window.

        number_of_nodes: int
            The number of nodes of the graph in the simulation.

        number_of_source_destination_pairs: int
            The number of nodes of the graph in the simulation.

        number_of_samples: int
            Number of times the simulation is repeated.
        """
        self.time_threshold = time_threshold
        self.original_capacity = original_capacity
        self.original_cost = original_cost
        self.long_link_cost = long_link_cost
        self.rebuild_probability = rebuild_probability
        self.number_of_nodes = number_of_nodes
        self.number_of_source_destination_pairs = number_of_source_destination_pairs
        self.number_of_samples = number_of_samples
        self.inf = float('inf')


def extract_argument(argument_dictionary: dict, key: str, default_value):
    """
    Helper function that extracts an argument from a dictionary of arguments.
    Returns a default value if the key is not in the dictionary.

    Parameters
    ----------
    argument_dictionary: dict
        Dictionary of arguments to be used in the simulation.

    key: str
        The key whose value is to be extracted from the dictionary of arguments.

    default_value
        Default value
    """
    if key not in argument_dictionary.keys():
        extracted_argument = default_value
    else:
        extracted_argument = argument_dictionary[key]
    return extracted_argument


def run_for_specific_source_destination_pair(number_of_source_destination_pairs: int, samples: int, algorithm=None,
                                             graph_edges: list = None, distance_threshold: int = None,
                                             propagation_radius: int = None, exponential_scale: bool = True,
                                             link_prediction: bool = False):

    results_for_source_destination = []
    link_length_dictionary = {}

    for x in range(1, samples + 1):

        # If the edges of the graph were not specified, then a random graph was specified
        if graph_edges is None:
            factory = graph_edge_factory.VirtualEdgeFactory(distance_threshold=distance_threshold)
            graph_edges = factory.generate_random_power_law_graph_edges()
            # helper.add_dictionary_to_dictionary(link_length_dictionary, link_lengths)

        if algorithm == routing_algorithms.local_knowledge_algorithm and propagation_radius is not None:
            results: tuple = algorithm(graph_edges, number_of_source_destination_pairs, propagation_radius,
                                       exponential_scale=exponential_scale)
        elif algorithm == routing_algorithms.global_knowledge_init:
            results: tuple = algorithm(graph_edges, number_of_source_destination_pairs,
                                       exponential_scale=exponential_scale)
        else:
            results: tuple = algorithm(graph_edges, number_of_source_destination_pairs, link_prediction=link_prediction,
                                       exponential_scale=exponential_scale)

        results_for_source_destination.append(results)
    return results_for_source_destination, link_length_dictionary


def extract_arguments_for_run_round(number_of_source_destination_pairs: int,
                                    samples: int, algorithm_arguments = None):
    """
    Execute the simulation for a specific number of source and destination pairs multiple times

    Parameters
    ----------
    number_of_source_destination_pairs: int
        Specifies the number of demands that need to be generated.

    samples: int
        The number of times the simulation will be repeated.

    algorithm_arguments: dict
        Dictionary containing the arguments needed for the algorithm.
    """

    # Extracting the arguments for the algorithm
    algorithm = extract_argument(algorithm_arguments, 'algorithm', routing_algorithms.initial_knowledge_init)
    graph_edges = extract_argument(algorithm_arguments, 'graph_edges', None)
    distance_threshold = extract_argument(algorithm_arguments, 'distance_threshold', 16)

    # If the algorithm is the local knowledge algorithm
    propagation_radius = extract_argument(algorithm_arguments, 'propagation_radius', None)

    link_prediction = extract_argument(algorithm_arguments, 'link_prediction', False)
    exponential_scale = extract_argument(algorithm_arguments, 'exponential_scale', False)

    # Running the simulation for the specific source and destination pairs
    results_for_source_destination, link_length_dictionary =\
        run_for_specific_source_destination_pair(number_of_source_destination_pairs, samples, algorithm, graph_edges,
                                                 distance_threshold, propagation_radius,
                                                 exponential_scale, link_prediction)

    return helper.map_tuple_gen(np.mean, zip(*results_for_source_destination)),\
           helper.map_tuple_gen(compute_mean_with_confidence, zip(*results_for_source_destination)),\
           link_length_dictionary


def compute_mean_with_confidence(data: list, confidence: float = 0.95):

    # Calculating confidence interval values of the result
    n = len(data)
    std_err = ss.sem(data)
    degrees_of_freedom = n - 1
    h = std_err * ss.t.ppf((1 + confidence) / 2, degrees_of_freedom)
    return h


def run_algorithm_for_graphs(number_of_source_destination_pairs: int, samples: int, algorithm_arguments = None):
    """
    Execute the simulation the specified number of source and destination pairs.

    Parameters
    ----------
    number_of_source_destination_pairs: int
        Specifies the number of demands that need to be generated.

    samples: int
        The number of times the simulation will be repeated.

    algorithm_arguments: dict
        Dictionary containing the arguments needed for the algorithm.
    """
    results_for_topology = []
    error_intervals_for_topology = []
    link_length_dictionary = {}

    # Iterating through the remaining simulation rounds
    for x in range(1, number_of_source_destination_pairs + 1):
        results, error, link_lengths = extract_arguments_for_run_round(x, samples, algorithm_arguments)

        # Summing up the values obtained in each round
        results_for_topology.append(results)
        error_intervals_for_topology.append(error)
        helper.add_dictionary_to_dictionary(link_length_dictionary, link_lengths)

    # Returning the average of the result values
    return tuple(list(result) for result in zip(*results_for_topology)),\
           tuple(list(result) for result in zip(*error_intervals_for_topology)), link_length_dictionary
