import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../lib")

import helper
import routing_algorithms
import graph_edge_factory
import numpy as np
from collections import defaultdict
import random
from typing import Callable, List
from enum import Enum


class LinkPredictionTypes(Enum):
    """Enumeration class to
    represent the type of
    centrality measures used for link prediction."""

    # Slightly different from betweenness centrality
    Betweenness = 1
    BetweennessCentrality = 2
    Closeness = 3


class AlgorithmSettings:
    def __init__(self, algorithm: Callable = None, propagation_radius: int = None,
                 link_prediction: LinkPredictionTypes = None, exponential_scale: bool = True):
        self.algorithm = algorithm
        self.propagation_radius = propagation_radius
        self.link_prediction = link_prediction
        self.exponential_scale = exponential_scale


class SimulationSettings:
    def __init__(self, number_of_source_destination_pairs: int = 50, number_of_samples: int = 1000):
        self.number_of_source_destination_pairs = number_of_source_destination_pairs
        self.number_of_samples = number_of_samples

    # TODO: implement constructor from dict
    '''
    @classmethod
    def from_dictionary_of_arguments(cls, arguments_as_dictionary):
        # Extracting the arguments for the algorithm
        algorithms = helper.extract_argument(arguments_as_dictionary, 'algorithm',
                                             routing_algorithms.initial_knowledge_init)

        # If the algorithm is the local knowledge algorithm
        propagation_radius = helper.extract_argument(arguments_as_dictionary, 'propagation_radius', None)

        link_prediction = helper.extract_argument(arguments_as_dictionary, 'link_prediction', False)
        exponential_scale = helper.extract_argument(arguments_as_dictionary, 'exponential_scale', False)

        simulation_settings = cls(algorithms, propagation_radius, link_prediction, exponential_scale)
        return simulation_settings
    '''


class TopologySettings:
    """
        Generic class for topology simulation settings.

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
    """
    def __init__(self, graph_edges: list = None, distance_threshold: int = 16, time_threshold: int = 1000,
                 original_capacity: int = 1, original_cost: int = 1, long_link_cost: int = 1000,
                 rebuild_probability: float = 0.25, number_of_nodes: int = 32):
        self.graph_edges = graph_edges
        self.distance_threshold = distance_threshold

        self.time_threshold = time_threshold
        self.original_capacity = original_capacity
        self.original_cost = original_cost
        self.long_link_cost = long_link_cost
        self.rebuild_probability = rebuild_probability
        self.number_of_nodes = number_of_nodes

    # TODO: implement constructor from dict
    '''
    @classmethod
    def from_dictionary_of_arguments(cls, arguments_as_dictionary):
        graph_edges = helper.extract_argument(arguments_as_dictionary, 'graph_edges', None)
        distance_threshold = helper.extract_argument(arguments_as_dictionary, 'distance_threshold', 16)

        topology_settings = cls(graph_edges, distance_threshold)
        return topology_settings
    '''


def map_tuple_gen(func, tup):
    return tuple(func(itup) for itup in tup)


def generate_random_source_destination(number_of_nodes: int) -> tuple:
    """
    Generates a random source and destination pair based on the number of nodes specified.

    Parameters
    ----------
    number_of_nodes : int
        Integer specifying the number of nodes in the graph.

    Returns
    -----
        Tuple containing the source and destination
    """
    random.seed()
    source = random.randint(1, number_of_nodes)
    dest = random.randint(1, number_of_nodes)
    while source == dest:
        dest = random.randint(1, number_of_nodes)
    return source, dest


def generate_random_pairs(number_of_pairs: int, number_of_nodes: int) -> list:
    """
    Generates a certain number of random source-destination pairs.

    Parameters
    ----------
    number_of_pairs : int
        Integer specifying the number of source-destination pairs to be generated.

    number_of_nodes : int
        Number of nodes used to generate random pairs.
    Returns
    -----
        List of tuples containing the source and destination nodes
    """
    result = []
    number_of_nodes = number_of_nodes

    for x in range(number_of_pairs):
        result.append(generate_random_source_destination(number_of_nodes))
    return result


class Simulation:
    """
    Generic class for simulation settings.

    Attributes
    ----------
    """
    def __init__(self, simulation_settings: SimulationSettings = None, topology_settings: TopologySettings = None,
                 list_of_algorithm_settings: List[AlgorithmSettings] = None):

        if simulation_settings is None:
            simulation_settings = SimulationSettings()

        self.simulation_settings = simulation_settings

        if topology_settings is None:
            topology_settings = TopologySettings()

        self.topology_settings = topology_settings

        if list_of_algorithm_settings is None:
            list_of_algorithm_settings = [AlgorithmSettings()]

        self.list_of_algorithm_settings = list_of_algorithm_settings

        self.current_results = defaultdict(list)
        self.intermediary_results = defaultdict(list)
        self.final_results = {}
        self.link_length_dictionary = defaultdict(list)
        self.errors = defaultdict(list)

    def run_algorithm_for_source_destination_pairs(self, algorithm_settings: AlgorithmSettings,
                                                   source_destination_pairs: list):

        name_of_approach = algorithm_settings.algorithm.__name__

        if algorithm_settings.link_prediction is not None:
            name_of_approach += algorithm_settings.link_prediction.name

        if not algorithm_settings.exponential_scale:
            name_of_approach += 'polynomial'

        if algorithm_settings.algorithm == routing_algorithms.local_knowledge_algorithm \
                and algorithm_settings.propagation_radius is not None:
            self.current_results[name_of_approach]\
                .append(algorithm_settings.algorithm(self.topology_settings.graph_edges, source_destination_pairs,
                                                     algorithm_settings.propagation_radius,
                                                     algorithm_settings.exponential_scale))
        elif algorithm_settings.algorithm == routing_algorithms.global_knowledge_init:
            self.current_results[name_of_approach]\
                .append(algorithm_settings.algorithm(self.topology_settings.graph_edges, source_destination_pairs,
                                                     algorithm_settings.exponential_scale))
        else:
            self.current_results[name_of_approach]\
                .append(algorithm_settings.algorithm(self.topology_settings.graph_edges, source_destination_pairs,
                                                     link_prediction=algorithm_settings.link_prediction,
                                                     exponential_scale= algorithm_settings.exponential_scale))

    def run_algorithms_several_times(self, number_of_source_destination_pairs: int):
        for x in range(1, self.simulation_settings.number_of_samples + 1):

            source_destination_pairs = generate_random_pairs(number_of_source_destination_pairs,
                                                             self.topology_settings.number_of_nodes)

            for current_algorithm_settings in self.list_of_algorithm_settings:
                self.run_algorithm_for_source_destination_pairs(current_algorithm_settings, source_destination_pairs)

    def extract_arguments_for_run_round(self, number_of_source_destination_pairs: int):
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

        # Running the simulation for the specific source and destination pairs
        self.run_algorithms_several_times(number_of_source_destination_pairs)

        # Get the point estimator for each of the samples
        for algorithm in self.current_results:
            self.intermediary_results[algorithm].append(map_tuple_gen(np.mean, zip(*self.current_results[algorithm])))
            self.current_results[algorithm].clear()

        # helper.map_tuple_gen(helper.compute_mean_with_confidence, zip(*results_for_source_destination))
        # link_length_dictionary

    def run_algorithm_for_graphs(self):
        """
        Execute the simulation the specified number of source and destination pairs.

        Parameters
        ----------
        """

        # Iterating through the remaining simulation rounds
        # TODO: loop here for the random graph 10 times
        # If the edges of the graph were not specified, then a random graph was specified

        if self.topology_settings.graph_edges is None:
            factory = graph_edge_factory.VirtualEdgeFactory(distance_threshold=
                                                            self.topology_settings.distance_threshold)
            self.topology_settings.graph_edges = factory.generate_random_power_law_graph_edges()

        for x in range(1, self.simulation_settings.number_of_source_destination_pairs + 1):
            # Generate random pairs of nodes between which we are seeking a path
            self.extract_arguments_for_run_round(x)

        for algorithm in self.intermediary_results:
            self.final_results[algorithm] = tuple(list(result) for result in zip(*self.intermediary_results[algorithm]))
        '''
            # Summing up the values obtained in each round
            results_for_topology.append(results)
            error_intervals_for_topology.append(error)
            helper.add_dictionary_to_dictionary(link_length_dictionary, link_lengths)

        # Returning the average of the result values
        return tuple(list(result) for result in zip(*results_for_topology)),\
               tuple(list(result) for result in zip(*error_intervals_for_topology)), link_length_dictionary
        '''
