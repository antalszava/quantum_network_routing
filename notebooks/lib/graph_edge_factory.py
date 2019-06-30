import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../lib")
import numpy as np
import graph


class VirtualEdgeFactory:
    """
    Class for a factory with which one can create the virtual edges of pre-defined virtual graphs.

    Attributes
    ----------
    number_of_nodes: int
        Number of nodes of the virtual graph.

    distance_threshold: int
        The distance threshold of the virtual graph.

    max_distance_threshold: int
        The maximum distance_threshold of the virtual graph.

    original_capacity: int
        Capacity of the edges to be created.

    alpha: int
        The parameter used for the construction of a random virtual graph.

    physical_graph_edges: list
        A list of edges making up the physical graph.

    physical_graph: Graph
        A physical graph constructed to create the edge list for the virtual graph.

    """

    def __init__(self, number_of_nodes: int = 32, distance_threshold: int = 1, max_distance_threshold: int = 16,
                 capacity: int = 1, alpha: int = 1):
        """
        The constructor for the VirtualEdgeFactory class.

        Parameters
        ----------
        number_of_nodes: int
            Number of nodes of the virtual graph.

        distance_threshold: int
            The distance threshold of the virtual graph.

        max_distance_threshold: int
            The maximum distance_threshold of the virtual graph.

        capacity: int
            Capacity of the edges to be created.

        alpha: int
            The parameter used for the construction of a random virtual graph.
        """
        self.number_of_nodes = number_of_nodes
        self.distance_threshold = distance_threshold
        self.max_distance_threshold = max_distance_threshold
        self.original_capacity = capacity
        self.alpha = alpha
        self.physical_graph_edges = [(x, x + 1, self.original_capacity) for x in range(1, self.number_of_nodes)]\
                                    + [(self.number_of_nodes, 1, self.original_capacity)]
        self.physical_graph = graph.Graph(self.physical_graph_edges)

    def shift_by_index(self, start: int, displacement: int) -> int:
        """
        Shifts the id of the vertex by a certain displacement (ring topology is assumed).
        """
        return (start + displacement - 1) % self.number_of_nodes + 1

    def get_choice_probability(self, start_node: int, end_node: int) -> float:
        """
        Returns the probability of choosing to create a link between the start and end vertex.

        Parameters
        ----------
        start_node: int
            Index of the starting vertex.
        end_node: int
            Index of the end vertex.
        """
        # Alternatively take out the end_node as well from the summing term
        beta_u = sum([1 / (self.physical_graph.physical_distance(start_node, x))
                      for x in range(1, self.number_of_nodes + 1) if x != start_node and
                      1 < self.physical_graph.physical_distance(start_node, x) <= self.distance_threshold])

        physical_distance = self.physical_graph.physical_distance(start_node, end_node)

        if 1 < physical_distance <= self.distance_threshold:
            prob = (1 / beta_u) * (1 / physical_distance)
            return prob
        else:
            return 0

    def power_law_link(self, start_node: int, long_link_length: int) -> int:
        """
        Returns the end node of a long link based on the result of sampling from a power law distribution.

        Parameters
        ----------
        start_node: int
            Index of the starting vertex.
        long_link_length: int
            Potential maximum length of the long link.
        """

        # These are the nodes that can be selected while sampling
        possible_nodes = {self.shift_by_index(start_node, x)
                          for x in range(-long_link_length, long_link_length + 1)}
        power_law_distribution = [self.get_choice_probability(start_node, x) for x in possible_nodes]

        probabilities = np.array(power_law_distribution)
        sum_power_np = probabilities.sum()

        probabilities /= sum_power_np
        np.random.seed()
        end_node = np.random.choice(list(possible_nodes), p=probabilities)

        return end_node

    def deterministic_link(self, start_node: int, long_link_length: int) -> int:
        """
        Returns the end node of a long link specified by the length of the long link.

        Parameters
        ----------
        start_node: int
            Index of the starting vertex.
        long_link_length: int
            Maximum length of the long link.
        """
        end_node = self.shift_by_index(start_node, long_link_length)
        return end_node

    def is_clockwise_indexed(self, start_node: int, end_node: int) -> bool:
        """
        Checks whether or not a start and end node are clockwise indexed.

        Parameters
        ----------
        start_node: int
            Index of the starting vertex.
        end_node: int
            Index of the end vertex.
        """
        return self.shift_by_index(start_node, self.physical_graph.physical_distance(start_node, end_node)) == end_node

    def reduce_edges(self, edge_list: list) -> list:
        """
        Reduces the same edges appearing in a list by increasing the capacity of the edge

        Parameters
        ----------
        edge_list: list
            List of edges to be reduced.
        """
        local_dictionary = {}
        local_list = []
        for (start, end, capacity) in edge_list:

            # Swap the start and end nodes, if not indexed in the same direction
            if not self.is_clockwise_indexed(start, end) or (abs(start-end) == self.distance_threshold and
                                                             self.distance_threshold == self.number_of_nodes/2
                                                             and 1 != start):
                start, end = end, start

            # Add them to the dictionary for keeping track
            if (start, end) in local_dictionary:
                local_dictionary[(start, end)] += capacity
            else:
                local_dictionary[(start, end)] = capacity
        for edge, capacity in local_dictionary.items():
            local_list.append((edge[0], edge[1], capacity))
        return local_list

    def generate_deterministic_virtual_link(self, distance_threshold: int, maximum_threshold: int = 16) -> list:
        """
        Reduces the same edges appearing in a list by increasing the capacity of the edge

        Parameters
        ----------
        distance_threshold: int
            The value of the distance threshold to be used for long link creation.
        maximum_threshold: int
            The value of the maximum distance threshold to be used for long link creation.

        Notes
        ----------
        We take the minimum of the distance threshold and the maximum threshold to create the long link.
        """
        long_link_length = min(distance_threshold, maximum_threshold)
        return [(x, self.deterministic_link(x, long_link_length),
                 self.original_capacity) for x in range(1, self.number_of_nodes) if x % maximum_threshold == 1]

    def generate_deterministic_graph_edges(self) -> list:
        """
        Generates the deterministic graph based on the original configuration of the graph edge factory.

        """
        virtual_links = []

        # Initiating the edges of 1) type: 1->3, 3->5, ... 31->1
        virtual_links += self.generate_deterministic_virtual_link(self.distance_threshold, 2)

        # Initiating the edges of 2) type: 1->5, 5->9, ... 29->1
        # Number of nodes have 0 as remainder for modulo 4
        virtual_links += self.generate_deterministic_virtual_link(self.distance_threshold, 4)

        if 8 <= self.max_distance_threshold:
            virtual_links += self.generate_deterministic_virtual_link(self.distance_threshold, 8)
        if 16 <= self.max_distance_threshold:
            virtual_links += self.generate_deterministic_virtual_link(self.distance_threshold, 16)

        # Add up the multiple links
        return self.reduce_edges(self.physical_graph_edges + virtual_links)

    def generate_random_power_law_graph_edges(self, number_of_links: int = 1) -> list:
        """
        Generates a random graph with a power law distribution based on the graph options specified and the
        number of links passed.

        Parameters
        ----------
        number_of_links: int
            Specifies how many long link each node will have.

        Raises
        ------

        Notes
        -----
            The distance_threshold specified for the graph creator needs to be at least 2 (otherwise there are no long
            links).

        """
        if 2 > self.distance_threshold:
            print('The distance threshold needs to be at least 2.')
        else:

            virtual_links = []

            for i in range(number_of_links):
                # Choose k many virtual links from each node in the graph
                virtual_links += [(x, self.power_law_link(x, self.distance_threshold),
                                   self.original_capacity) for x in range(1, self.number_of_nodes + 1)]

            # Add up the multiple links
            final_virtual_graph = self.physical_graph_edges + virtual_links
            return self.reduce_edges(final_virtual_graph)
