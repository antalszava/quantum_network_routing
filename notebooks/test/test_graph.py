import unittest

import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../lib")
import routing_algorithms
import graph
import graph_edge_factory


class TestGetEdgeFrequencies(unittest.TestCase):

    """"
    get_frequency_for_path
    """
    def test_empty_dict_get_frequency_for_path(self):
        """
        Testing
        (a) Clearing the edge frequencies after instantiation of the graph
        (b) Check if the return value is None

        """
        path_length = 50
        local_current_path_list = [x for x in range(path_length)]

        factory = graph_edge_factory.GraphEdgesFactory(distance_threshold=1)
        graph_edges = factory.generate_deterministic_graph_edges(factory.deterministic_link)
        local_graph = graph.Graph(graph_edges)

        # Clear the edge_frequencies
        local_graph.edge_frequencies = {}

        self.assertEqual(local_graph.add_frequency_for_path(local_current_path_list), None)
        self.assertEqual(local_graph.edge_frequencies, {(x, x+1): 1 for x in range(path_length-1)})

    """"
    get_paths_for_all_pairs
    """
    def test_number_of_paths_get_paths_for_all_pairs(self):
        """
        Testing
        (a) the number of paths created
        (b) if all pairs are tested
        """
        for dth in range(1, 5):

            # Creating the related graph object
            threshold = 2 ** dth

            factory = graph_edge_factory.GraphEdgesFactory(distance_threshold=threshold)
            deterministic_edges = factory.generate_deterministic_graph_edges(factory.deterministic_link)
            local_graph = graph.Graph(deterministic_edges)

            number_of_nodes = len(local_graph.Vertices)
            expected_number_of_pairs = number_of_nodes*(number_of_nodes-1)

            # Generating the paths
            local_paths = local_graph.get_paths_for_all_pairs()

            self.assertEqual(len(local_paths), expected_number_of_pairs)
            list_of_pairs = [(path[-1:][0], path[:1][0]) for path in local_paths]

            # See if we really have the same source-destination pairs
            self.assertEqual(set(list_of_pairs), set([(x, y) for x in range(1,number_of_nodes+1)
                                                      for y in range(1, number_of_nodes+1) if x != y]))

    """"
    get_edge_frequencies_in_graph
    """
    def test_simple_graph_get_edge_frequencies_in_graph(self):
        """
        Testing
        (a) the frequency for a simple graph
        (b) the number of edges in the dictionary and in the paths generated
        """
        graph_edges = [(1, 2, 0), (2, 3, 0), (3, 1, 0)]
        local_graph = graph.Graph(graph_edges, link_prediction=True)

        number_of_edges = sum([len(path)-1 for path in local_graph.get_paths_for_all_pairs()])

        local_frequencies = local_graph.edge_frequencies

        expected_frequencies = {(1, 2): 2, (1, 3): 2, (2,3): 2}

        self.assertEqual(local_frequencies, expected_frequencies)

        # Check if we really have the right number of edges
        self.assertEqual(number_of_edges, sum(expected_frequencies.values()))

    def test_complex_graphs_get_edge_frequencies_in_graph(self):
        """
        Testing the number of edges in the dictionary and in the paths generated
        """
        for dth in range(0, 5):
            threshold = 2 ** dth

            # Creating on-demand graphs
            factory = graph_edge_factory.GraphEdgesFactory(distance_threshold=threshold, original_capacity=0)
            graph_edges = factory.generate_deterministic_graph_edges(factory.deterministic_link)
            # print(graph_edges)
            temp_graph = graph.Graph(graph_edges, link_prediction=True)

            number_of_edges = sum([len(path) - 1 for path in temp_graph.get_paths_for_all_pairs()])

            local_frequencies = temp_graph.edge_frequencies

            # Check if we really have the right number of edges
            self.assertEqual(number_of_edges, sum(local_frequencies.values()))


if __name__ == '__main__':
    unittest.main()
