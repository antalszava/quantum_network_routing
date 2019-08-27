import unittest


import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../lib")
import routing_algorithms
import routing_simulation
import graph_edge_factory
import graph
import shortest_path
import networkx as nx


class TestDijkstra(unittest.TestCase):
    def test_dth1_simple_path(self):
        """
        Test that it can sum a list of integers
        """
        factory = graph_edge_factory.VirtualEdgeFactory()
        graph_edges = factory.generate_deterministic_graph_edges()
        local_graph = graph.Graph(graph_edges)

        self.assertEqual(shortest_path.dijkstra(local_graph, 1, 3), [3,2,1])

    def test_dth2_simple_path(self):
        """
        Test that it can sum a list of integers
        """
        factory = graph_edge_factory.VirtualEdgeFactory(distance_threshold=2)
        graph_edges = factory.generate_deterministic_graph_edges()
        local_graph = graph.Graph(graph_edges)

        self.assertEqual(shortest_path.dijkstra(local_graph, 1, 3), [3, 1])

    def test_dth4_simple_path(self):
        """
        Test that it can sum a list of integers
        """
        factory = graph_edge_factory.VirtualEdgeFactory(distance_threshold=4)
        graph_edges = factory.generate_deterministic_graph_edges()
        local_graph = graph.Graph(graph_edges)

        self.assertEqual(shortest_path.dijkstra(local_graph, 1, 5), [5, 1])

    def test_dth1_complex_path(self):
        """
        Test that it can sum a list of integers
        """
        factory = graph_edge_factory.VirtualEdgeFactory()
        graph_edges = factory.generate_deterministic_graph_edges()
        local_graph = graph.Graph(graph_edges)

        self.assertEqual(shortest_path.dijkstra(local_graph, 1, 20), [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                                                           31, 32, 1])

    def test_dth2_complex_path(self):
        """
        Test that it can sum a list of integers
        """
        factory = graph_edge_factory.VirtualEdgeFactory(distance_threshold=2)
        graph_edges = factory.generate_deterministic_graph_edges()
        local_graph = graph.Graph(graph_edges)

        self.assertEqual(shortest_path.dijkstra(local_graph, 1, 20), [20, 21, 23, 25, 27, 29, 31, 1])

    def test_dth4_complex_path(self):
        """
        Test that it can sum a list of integers
        """
        factory = graph_edge_factory.VirtualEdgeFactory(distance_threshold=4)
        graph_edges = factory.generate_deterministic_graph_edges()
        local_graph = graph.Graph(graph_edges)

        self.assertEqual(shortest_path.dijkstra(local_graph, 1, 20), [20, 21, 25, 29, 1])

    def test_dth2_long_links_path(self):
        """
        Test that it can sum a list of integers
        """
        factory = graph_edge_factory.VirtualEdgeFactory(distance_threshold=2)
        graph_edges = factory.generate_deterministic_graph_edges()
        local_graph = graph.Graph(graph_edges)

        self.assertEqual(shortest_path.dijkstra(local_graph, 1, 15), [15, 13, 11, 9, 7, 5, 3, 1])

    def test_dth4_long_links_path(self):
        """
        Test that it can sum a list of integers
        """
        factory = graph_edge_factory.VirtualEdgeFactory(distance_threshold=4, max_distance_threshold=4)
        graph_edges = factory.generate_deterministic_graph_edges()
        local_graph = graph.Graph(graph_edges)

        # self.assertEqual(shortest_path.dijkstra(local_graph, 1, 17), [17, 13, 9, 5, 1])
        self.assertEqual(shortest_path.dijkstra(local_graph, 1, 27), [27, 29, 1])
        self.assertEqual(shortest_path.dijkstra(local_graph, 1, 26), [26, 27, 29, 1])
        self.assertEqual(shortest_path.dijkstra(local_graph, 1, 25), [25, 29, 1])
        self.assertEqual(shortest_path.dijkstra(local_graph, 1, 24), [24, 25, 29, 1])
        self.assertEqual(shortest_path.dijkstra(local_graph, 1, 23), [23, 25, 29, 1])
        self.assertEqual(shortest_path.dijkstra(local_graph, 1, 22), [22, 23, 25, 29, 1])
        self.assertEqual(shortest_path.dijkstra(local_graph, 1, 21), [21, 25, 29, 1])
        self.assertEqual(shortest_path.dijkstra(local_graph, 1, 20), [20, 21, 25, 29, 1])
        self.assertEqual(shortest_path.dijkstra(local_graph, 1, 19), [19, 21, 25, 29, 1])
        self.assertEqual(shortest_path.dijkstra(local_graph, 1, 18), [18, 17, 13, 9, 5, 1])
        self.assertEqual(shortest_path.dijkstra(local_graph, 16, 1), [1, 29, 25, 21, 17, 16])
        self.assertEqual(shortest_path.dijkstra(local_graph, 1, 17), [17, 13, 9, 5, 1])
        self.assertEqual(shortest_path.dijkstra(local_graph, 1, 16), [16, 17, 13, 9, 5, 1])
        self.assertEqual(shortest_path.dijkstra(local_graph, 1, 15), [15, 13, 9, 5, 1])
        self.assertEqual(shortest_path.dijkstra(local_graph, 1, 14), [14, 13, 9, 5, 1])

    def test_dth8_long_links_path(self):
        """
        Test that it can sum a list of integers
        """
        factory = graph_edge_factory.VirtualEdgeFactory(distance_threshold=8)
        graph_edges = factory.generate_deterministic_graph_edges()
        local_graph = graph.Graph(graph_edges)

        # self.assertEqual(shortest_path.dijkstra(local_graph, 1, 17), [17, 13, 9, 5, 1])
        self.assertEqual(shortest_path.dijkstra(local_graph, 1, 19), [19, 17, 9, 1])
        self.assertEqual(shortest_path.dijkstra(local_graph, 19, 1), [1, 9, 17, 19])
        self.assertEqual(shortest_path.dijkstra(local_graph, 1, 18), [18, 17, 9, 1])
        self.assertEqual(shortest_path.dijkstra(local_graph, 18, 1), [1, 9, 17, 18])
        self.assertEqual(shortest_path.dijkstra(local_graph, 1, 16), [16, 17, 9, 1])
        self.assertEqual(shortest_path.dijkstra(local_graph, 1, 15), [15, 17, 9, 1])
        self.assertEqual(shortest_path.dijkstra(local_graph, 1, 14), [14, 13, 9, 1])

    def test_dth16_long_links_path(self):
        """
        Test that it can sum a list of integers
        """
        factory = graph_edge_factory.VirtualEdgeFactory(distance_threshold=16)
        graph_edges = factory.generate_deterministic_graph_edges()
        local_graph = graph.Graph(graph_edges)

        self.assertEqual(shortest_path.dijkstra(local_graph, 1, 17), [17, 1])

        self.assertTrue(shortest_path.dijkstra(local_graph, 6, 18) != [18, 17, 1, 5, 6])

        self.assertEqual(shortest_path.dijkstra(local_graph, 32, 18), [18, 17, 1, 32])
        self.assertEqual(shortest_path.dijkstra(local_graph, 31, 18), [18, 17, 1, 31])
        self.assertEqual(shortest_path.dijkstra(local_graph, 31, 18), [18, 17, 1, 31])
        self.assertEqual(shortest_path.dijkstra(local_graph, 30, 18), [18, 17, 1, 29, 30])
        self.assertEqual(shortest_path.dijkstra(local_graph, 29, 18), [18, 17, 1, 29])
        self.assertEqual(shortest_path.dijkstra(local_graph, 28, 18), [18, 17, 1, 29, 28])

        self.assertTrue(shortest_path.dijkstra(local_graph, 27, 18) != [18, 17, 1, 29, 27])
        self.assertTrue(shortest_path.dijkstra(local_graph, 6, 18) != [18, 17, 1, 5, 6])

    def test_dth4_on_demand_long_links_path(self):
        """
        Test that it can sum a list of integers
        """
        factory = graph_edge_factory.VirtualEdgeFactory(distance_threshold=4, capacity=0)
        graph_edges = factory.generate_deterministic_graph_edges()
        local_graph = graph.Graph(graph_edges)

        # self.assertEqual(shortest_path.dijkstra(local_graph, 1, 17), [17, 13, 9, 5, 1])
        self.assertEqual(shortest_path.dijkstra(local_graph, 1, 16), [16, 15, 13, 9, 5, 1])
        self.assertEqual(shortest_path.dijkstra(local_graph, 16, 1), [1, 5, 9, 13, 15, 16])
        self.assertEqual(shortest_path.dijkstra(local_graph, 1, 15), [15, 13, 9, 5, 1])
        self.assertEqual(shortest_path.dijkstra(local_graph, 15, 1), [1, 5, 9, 13, 15])
        self.assertEqual(shortest_path.dijkstra(local_graph, 1, 14), [14, 13, 9, 5, 1])
        self.assertEqual(shortest_path.dijkstra(local_graph, 14, 1), [1, 5, 9, 13, 14])

    def test_dth8_on_demand_long_links_path(self):
        """
        Test that it can sum a list of integers
        """
        factory = graph_edge_factory.VirtualEdgeFactory(distance_threshold=8, capacity=0)
        graph_edges = factory.generate_deterministic_graph_edges()
        local_graph = graph.Graph(graph_edges)

        # self.assertEqual(shortest_path.dijkstra(local_graph, 1, 17), [17, 13, 9, 5, 1])
        self.assertEqual(shortest_path.dijkstra(local_graph, 1, 19), [19, 21, 25, 1])

        # Disclaimer: the following is longer than the expected shortest path, as we are in the on-demand model
        self.assertEqual(shortest_path.dijkstra(local_graph, 1, 18), [18, 19, 21, 25, 1])
        self.assertEqual(shortest_path.dijkstra(local_graph, 1, 16), [16, 15, 13, 9, 1])
        self.assertEqual(shortest_path.dijkstra(local_graph, 1, 15), [15, 13, 9, 1])
        self.assertEqual(shortest_path.dijkstra(local_graph, 1, 14), [14, 13, 9, 1])

    def test_node_index_non_existent(self):
        """
        Test that it can sum a list of integers
        """
        factory = graph_edge_factory.VirtualEdgeFactory(distance_threshold=4)
        graph_edges = factory.generate_deterministic_graph_edges()
        local_graph = graph.Graph(graph_edges)

        self.assertRaises(KeyError, shortest_path.dijkstra, local_graph, 33, 35)

    def test_same_source_destination(self):
        """
        Test that it can sum a list of integers
        """
        factory = graph_edge_factory.VirtualEdgeFactory(distance_threshold=4)
        graph_edges = factory.generate_deterministic_graph_edges()
        local_graph = graph.Graph(graph_edges)

        for x in range(1, factory.number_of_nodes+1):
            self.assertEqual(shortest_path.dijkstra(local_graph, x, x), [x])

    def test_always_same_as_nx(self):
        factory = graph_edge_factory.VirtualEdgeFactory(distance_threshold=2, max_distance_threshold=16)
        graph_edges = factory.generate_deterministic_graph_edges()
        local_graph = graph.Graph(graph_edges)
        nx_graph = nx.Graph()
        for x in graph_edges:
            nx_graph.add_edge(x[0], x[1])

        for x in range(1, factory.number_of_nodes + 1):
            for y in range(1, factory.number_of_nodes + 1):
                if x != y:
                    shortest_path1 = shortest_path.dijkstra(local_graph, x, y)
                    shortest_path2 = nx.shortest_path(nx_graph, x, y)
                    self.assertEqual(len(shortest_path1), len(shortest_path2))
                    self.assertTrue(shortest_path1[0] == y)
                    self.assertTrue(shortest_path1[len(shortest_path1)-1] == x)
                    index = 0
                    for node in shortest_path1:
                        self.assertEqual(node, shortest_path1[index])
                        index += 1
                        if index == len(shortest_path1)-2:
                            self.assertTrue(shortest_path1[index+1] in nx.neighbors(nx_graph, shortest_path1[index]))


class TestLinkPredictionDijkstra(unittest.TestCase):
    def test_dth2_simple_path(self):
        """
        Test that it computes the paths according to the local data
        """
        link_prediction = True
        factory = graph_edge_factory.VirtualEdgeFactory(distance_threshold=2)
        graph_edges = factory.generate_deterministic_graph_edges()
        local_graph = graph.Graph(graph_edges, link_prediction=link_prediction)

        # Shortest paths in the virtual graph
        self.assertEqual(shortest_path.dijkstra(local_graph, 1, 3, link_prediction=True), [3, 1])
        self.assertEqual(shortest_path.dijkstra(local_graph, 1, 5, link_prediction=True), [5, 3, 1])

        current_step = 1000
        local_graph.update_stored_weights(current_step)\

        # Still getting the same result, as 1 is equal to the source -> knows the availability of the 1-3 link
        self.assertEqual(shortest_path.dijkstra(local_graph, 1, 3, link_prediction=True), [3, 1])

        # Link 3-5 is far away, so we move along the physical graph
        self.assertEqual(shortest_path.dijkstra(local_graph, 1, 5, link_prediction=True), [5, 4, 3, 1])

    def test_dth4_simple_path(self):
        """
        Test that it computes the paths according to the local data
        """
        link_prediction = True
        factory = graph_edge_factory.VirtualEdgeFactory(distance_threshold=4)
        graph_edges = factory.generate_deterministic_graph_edges()
        local_graph = graph.Graph(graph_edges, link_prediction=link_prediction)

        # Shortest paths in the virtual graph
        self.assertEqual(shortest_path.dijkstra(local_graph, 1, 3, link_prediction=True), [3, 1])
        self.assertEqual(shortest_path.dijkstra(local_graph, 1, 5, link_prediction=True), [5, 1])
        self.assertEqual(shortest_path.dijkstra(local_graph, 1, 9, link_prediction=True), [9, 5, 1])

        current_step = 1000
        local_graph.update_stored_weights(current_step)\

        # Still getting the same result, as 1 is equal to the source -> knows the availability of the 1-3 link
        self.assertEqual(shortest_path.dijkstra(local_graph, 1, 3, link_prediction=True), [3, 1])
        self.assertEqual(shortest_path.dijkstra(local_graph, 1, 5, link_prediction=True), [5, 1])

        # Link 5-9 is far away, so we move along the physical graph
        self.assertEqual(shortest_path.dijkstra(local_graph, 1, 9, link_prediction=True), [9, 8, 7, 6, 5, 1])

    def test_always_same_as_nx(self):
        link_prediction = True
        factory = graph_edge_factory.VirtualEdgeFactory(distance_threshold=2, max_distance_threshold=16)
        graph_edges = factory.generate_deterministic_graph_edges()
        local_graph = graph.Graph(graph_edges, link_prediction)
        nx_graph = nx.Graph()
        for x in graph_edges:
            nx_graph.add_edge(x[0], x[1])

        for x in range(1, factory.number_of_nodes + 1):
            for y in range(1, factory.number_of_nodes + 1):
                if x != y:
                    shortest_path1 = shortest_path.dijkstra(local_graph, x, y, link_prediction=link_prediction)
                    shortest_path2 = nx.shortest_path(nx_graph, x, y)
                    self.assertEqual(len(shortest_path1), len(shortest_path2))
                    self.assertTrue(shortest_path1[0] == y)
                    self.assertTrue(shortest_path1[len(shortest_path1) - 1] == x)
                    index = 0
                    for node in shortest_path1:
                        self.assertEqual(node, shortest_path1[index])
                        index += 1
                        if index == len(shortest_path1) - 2:
                            self.assertTrue(shortest_path1[index + 1] in nx.neighbors(nx_graph, shortest_path1[index]))


if __name__ == '__main__':
    unittest.main()
