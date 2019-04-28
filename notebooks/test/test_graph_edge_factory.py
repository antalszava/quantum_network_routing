import unittest

import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../lib")

import graph_edge_factory
import math


class TestGraphEdgesFactory(unittest.TestCase):
    def test_shift_by_index(self):
        for number in range(1, 50):
            factory = graph_edge_factory.GraphEdgesFactory(number_of_nodes = number)
            for x in range(0, 200):
                for y in range(0, 200):
                    if (x + y) % factory.number_of_nodes == 0:
                        self.assertEqual(factory.shift_by_index(x, y), factory.number_of_nodes)
                    else:
                        self.assertEqual(factory.shift_by_index(x, y), (x + y) % factory.number_of_nodes)
    # TODO: test reduce edges
    # def test_reduce_edges(self):


class TestDeterministic(unittest.TestCase):
    def test_sum_of_edges_deterministic_graph(self):
        """
        Test that it can sum a list of integers
        """
        for dth in range(0, 5):
            threshold = 2 ** dth
            factory = graph_edge_factory.GraphEdgesFactory(distance_threshold=threshold, max_threshold=4)
            graph_edges = factory.generate_deterministic_graph_edges(factory.deterministic_link)

            # Checking if the number of edges is the same as expected
            sum_of_edges = sum([x[2] for x in graph_edges])

            number_of_edges = sum([int(factory.number_of_nodes/2**x) for x in range(0,
                                                                        int(math.log(factory.max_threshold, 2)) + 1)])

            self.assertEqual(sum_of_edges, number_of_edges, "The number of edges is not the expected value.")


'''
Testing the randomly generated power law graph
'''


class TestRandomPowerLaw(unittest.TestCase):
    def test_sum_of_edges(self):
        """
        Test that it can sum a list of integers
        """
        for x in range(100):
            max_power = 4
            for dth in range(1, max_power+1):
                threshold = 2**dth
                factory = graph_edge_factory.GraphEdgesFactory(distance_threshold=threshold, max_threshold=2**max_power)
                graph_edges, l = factory.generate_random_power_law_graph_edges()

                # Number of neighbours that are selected by each node
                k = 1
                    #int(math.log(factory.max_threshold, 2))

                # Checking if the number of edges is the same as expected
                sum_of_edges = sum([x[2] for x in graph_edges])

                number_of_edges = factory.number_of_nodes + factory.number_of_nodes*k

                self.assertEqual(sum_of_edges, number_of_edges, "The number of edges is not the expected value.")

    '''
    Helper functions
    '''

    @staticmethod
    def get_link_length(start_node: int, end_node: int, number_of_nodes: int):
        return (end_node - start_node) if start_node < end_node else (number_of_nodes - start_node + end_node)

    @staticmethod
    def are_long_neighbours(start_node: int, end_node: int, number_of_nodes: int):
        return abs(start_node - end_node) > 1 and (start_node != number_of_nodes or end_node != 1)

    def test_long_links_within_threshold(self):
        """
        Test that long links are indeed within a certain threshold
        """
        for dth in range(1, 4):
            threshold = 2**dth
            factory = graph_edge_factory.GraphEdgesFactory(distance_threshold=threshold)
            graph_edges, l = factory.generate_random_power_law_graph_edges()

            count = 0
            for edge in graph_edges:

                # Checking for non-neighboring nodes:
                if self.are_long_neighbours(edge[0], edge[1], factory.number_of_nodes):
                    link_length = self.get_link_length(edge[0], edge[1], factory.number_of_nodes)

                    self.assertTrue(1 < link_length <= threshold,
                                    "The length of long links should be greater than one but smaller than or equal to"
                                    "the threshold value.")
                    count += edge[2]

            number_of_edges = factory.number_of_nodes + factory.number_of_nodes * \
                              int(math.log(factory.max_threshold, 2))

            # We have checked all the edges
            self.assertEqual(count, number_of_edges-factory.number_of_nodes, "There is a difference in the number of"
                                                                " edges checked and the number of edges in the graph.")


'''
Testing the deterministically generated power law graph
'''


class TestPowerLaw(unittest.TestCase):
    def test_sum_of_edges(self):
        """
        Test that it can sum a list of integers
        """

        for dth in range(1, 4):
            factory = graph_edge_factory.GraphEdgesFactory(distance_threshold=2**dth)
            graph_edges = factory.generate_deterministic_graph_edges(factory.power_law_link)

            # Checking if the number of edges is the same as expected
            sum_of_edges = sum([x[2] for x in graph_edges])

            number_of_edges = sum([factory.number_of_nodes/ 2 ** x
                                   for x in range(0, int(math.log(factory.max_threshold, 2)) + 1)])

            self.assertEqual(sum_of_edges, number_of_edges, "The number of edges is not the expected value.")

    def test_neighbouring_link_is_unique(self):
        """
        Test that neighbours only share one virtual link
        """

        for dth in range(1, 3):
            factory = graph_edge_factory.GraphEdgesFactory(distance_threshold=2**dth)
            graph_edges = factory.generate_deterministic_graph_edges(factory.power_law_link)

            # Checking if neighbouring nodes only have exactly one link
            count = 0
            for edge in graph_edges:

                # Checking for neighboring nodes:
                # (A) Either they have consequtive indices
                # (B) Or we are already at the end of the cycle and have the last edge
                if abs(edge[0]-edge[1]) == 1 or (edge[0] == factory.number_of_nodes and edge[1] == 1):
                    self.assertEqual(edge[2], 1, "The should be only one edge between neighbouring nodes.")
                    count += 1

            # We have checked all the nodes
            self.assertEqual(count, factory.number_of_nodes, "There is a difference in the number of nodes checked and "
                                                             "the number of nodes in the graph.")

    @staticmethod
    def get_link_length(start_node: int, end_node: int, number_of_nodes: int):
        return (end_node - start_node) if start_node < end_node else (number_of_nodes - start_node + end_node)

    @staticmethod
    def are_long_neighbours(start_node: int, end_node: int, number_of_nodes: int):
        return abs(start_node - end_node) > 1 and (start_node != number_of_nodes or end_node != 1)

    def test_long_links_within_threshold(self):
        """
        Test that long links are indeed within a certain threshold
        """
        for dth in range(1, 3):
            threshold = 2**dth
            factory = graph_edge_factory.GraphEdgesFactory(distance_threshold=threshold)
            graph_edges = factory.generate_deterministic_graph_edges(factory.power_law_link)

            count = 0
            for edge in graph_edges:

                # Checking for non-neighboring nodes:
                if self.are_long_neighbours(edge[0], edge[1], factory.number_of_nodes):
                    link_length = self.get_link_length(edge[0], edge[1], factory.number_of_nodes)

                    self.assertTrue(1 < link_length <= threshold,
                                    "The length of long links should be greater than one but smaller than or equal to"
                                    "the threshold value.")
                    count += edge[2]

            number_of_edges = sum([factory.number_of_nodes/ 2 ** x
                                   for x in range(0, int(math.log(factory.max_threshold, 2)) + 1)])

            # We have checked all the edges
            self.assertEqual(count, number_of_edges-factory.number_of_nodes, "There is a difference in the number of"
                                                                " edges checked and the number of edges in the graph.")

    def test_long_link_source_is_correct(self):
        for dth in range(1, 3):
            factory = graph_edge_factory.GraphEdgesFactory(distance_threshold=2**dth)
            graph_edges = factory.generate_deterministic_graph_edges(factory.power_law_link)

            long_link_source = [x for x in range(1, factory.number_of_nodes + 1)
                                if x % 2 == 1 and x < factory.number_of_nodes]

            long_link_source += [x for x in range(1, factory.number_of_nodes)
                                 if x % 4 == 1]

            for edge in graph_edges:
                # Checking for non-neighboring nodes:

                if abs(edge[0] - edge[1]) != 1 and (edge[0] != factory.number_of_nodes) and edge[1] != 1:
                    self.assertTrue(edge[0] in long_link_source or edge[1] in long_link_source,
                                    "A long link was created with the wrong source")


if __name__ == '__main__':
    unittest.main()