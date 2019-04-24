import unittest


import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../lib")
import routing_algorithms
import graph_edge_factory
import graph

class TestTracebackPath(unittest.TestCase):
    def test_list_int(self):
        """
        Test that it can sum a list of integers
        """
        data = {3: None, 4: 3, 7: 4, 12: 16, 16: 7, 2: 12}
        result = routing_algorithms.traceback_path(2, data)
        self.assertEqual(result, [2, 12, 16, 7, 4, 3])

    def test_end_of_path_not_destination(self):
        """
        Test that when tracing back the path, then we don't end up at the destination (that would be 16)
        """
        data = {16: None, 7: 3, 12: 7, 2: 12}
        self.assertRaises(KeyError, routing_algorithms.traceback_path, 2, data)


class TestEntanglementSwap(unittest.TestCase):
    def test_on_demand_graph(self):

        factory = graph_edge_factory.GraphEdgesFactory(distance_threshold=1, original_capacity=0)
        graph_edges = factory.generate_deterministic_graph_edges(factory.deterministic_link)
        main_graph = graph.Graph(graph_edges)

        # For each of the edge, we don't increase the waiting time, but increment the number of no-links
        for (start, end, capacity) in graph_edges:
            self.assertEqual(capacity, 0)
            self.assertEqual(routing_algorithms.entanglement_swap(main_graph, start, end), (0, 1))

    def test_dth1(self):

        factory = graph_edge_factory.GraphEdgesFactory(distance_threshold=1, original_capacity=1)
        graph_edges = factory.generate_deterministic_graph_edges(factory.deterministic_link)
        main_graph = graph.Graph(graph_edges)

        # For each of the edge, we increment the waiting time, but not the number of no-links
        for (start, end, capacity) in graph_edges:
            for x in range(capacity):
                self.assertEqual(main_graph.get_edge_capacity(start, end), capacity-x)
                self.assertEqual(routing_algorithms.entanglement_swap(main_graph, start, end), (1, 0))


class TestDijkstra(unittest.TestCase):
    def test_dth1_simple_path(self):
        """
        Test that it can sum a list of integers
        """
        factory = graph_edge_factory.GraphEdgesFactory()
        graph_edges = factory.generate_deterministic_graph_edges(factory.deterministic_link)
        local_graph = graph.Graph(graph_edges)

        self.assertEqual(routing_algorithms.dijkstra(local_graph, 1, 3), [3,2,1])

    def test_dth2_simple_path(self):
        """
        Test that it can sum a list of integers
        """
        factory = graph_edge_factory.GraphEdgesFactory(distance_threshold=2)
        graph_edges = factory.generate_deterministic_graph_edges(factory.deterministic_link)
        local_graph = graph.Graph(graph_edges)

        self.assertEqual(routing_algorithms.dijkstra(local_graph, 1, 3), [3, 1])

    def test_dth4_simple_path(self):
        """
        Test that it can sum a list of integers
        """
        factory = graph_edge_factory.GraphEdgesFactory(distance_threshold=4)
        graph_edges = factory.generate_deterministic_graph_edges(factory.deterministic_link)
        local_graph = graph.Graph(graph_edges)

        self.assertEqual(routing_algorithms.dijkstra(local_graph, 1, 5), [5, 1])

    def test_dth1_complex_path(self):
        """
        Test that it can sum a list of integers
        """
        factory = graph_edge_factory.GraphEdgesFactory()
        graph_edges = factory.generate_deterministic_graph_edges(factory.deterministic_link)
        local_graph = graph.Graph(graph_edges)

        self.assertEqual(routing_algorithms.dijkstra(local_graph, 1, 20), [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                                                           31, 32, 1])

    def test_dth2_complex_path(self):
        """
        Test that it can sum a list of integers
        """
        factory = graph_edge_factory.GraphEdgesFactory(distance_threshold=2)
        graph_edges = factory.generate_deterministic_graph_edges(factory.deterministic_link)
        local_graph = graph.Graph(graph_edges)

        self.assertEqual(routing_algorithms.dijkstra(local_graph, 1, 20), [20, 21, 23, 25, 27, 29, 31, 1])

    def test_dth4_complex_path(self):
        """
        Test that it can sum a list of integers
        """
        factory = graph_edge_factory.GraphEdgesFactory(distance_threshold=4)
        graph_edges = factory.generate_deterministic_graph_edges(factory.deterministic_link)
        local_graph = graph.Graph(graph_edges)

        self.assertEqual(routing_algorithms.dijkstra(local_graph, 1, 20), [20, 21, 25, 29, 1])

    def test_dth2_long_links_path(self):
        """
        Test that it can sum a list of integers
        """
        factory = graph_edge_factory.GraphEdgesFactory(distance_threshold=2)
        graph_edges = factory.generate_deterministic_graph_edges(factory.deterministic_link)
        local_graph = graph.Graph(graph_edges)

        self.assertEqual(routing_algorithms.dijkstra(local_graph, 1, 15), [15, 13, 11, 9, 7, 5, 3, 1])


    def test_dth4_long_links_path(self):
        """
        Test that it can sum a list of integers
        """
        factory = graph_edge_factory.GraphEdgesFactory(distance_threshold=4)
        graph_edges = factory.generate_deterministic_graph_edges(factory.deterministic_link)
        local_graph = graph.Graph(graph_edges)

        self.assertEqual(routing_algorithms.dijkstra(local_graph, 1, 17), [17, 13, 9, 5, 1])

    def test_node_index_non_existent(self):
        """
        Test that it can sum a list of integers
        """
        factory = graph_edge_factory.GraphEdgesFactory(distance_threshold=4)
        graph_edges = factory.generate_deterministic_graph_edges(factory.deterministic_link)
        local_graph = graph.Graph(graph_edges)

        self.assertRaises(KeyError, routing_algorithms.dijkstra, local_graph, 33, 35)

    def test_same_source_destination(self):
        """
        Test that it can sum a list of integers
        """
        factory = graph_edge_factory.GraphEdgesFactory(distance_threshold=4)
        graph_edges = factory.generate_deterministic_graph_edges(factory.deterministic_link)
        local_graph = graph.Graph(graph_edges)

        for x in range(1, factory.number_of_nodes+1):
            self.assertEqual(routing_algorithms.dijkstra(local_graph, x, x), [x])

'''
Functions called upon in the local knowledge method: create_graph_with_local_knowledge, dijkstra,
                                                    distribute_entanglement, update_local_knowledge 
'''
class TestLocalKnowledge(unittest.TestCase):

    @staticmethod
    def does_node_have_correct_knowledge(graph_edges: list, local_knowledge):
        return True

    def test_create_graph_(self):
        for dth in range(1, 5):
            threshold = 2 ** dth
            factory = graph_edge_factory.GraphEdgesFactory(distance_threshold=threshold)


            # Check for deterministic graph
            deterministic_edges = factory.generate_deterministic_graph_edges(factory.deterministic_link)
            local_knowledge_graph = routing_algorithms.create_graph_with_local_knowledge(deterministic_edges)

            main_graph = graph.Graph(deterministic_edges)
            for vertex in local_knowledge_graph.Vertices.values():
                self.assertEqual(vertex.local_knowledge, main_graph)


if __name__ == '__main__':
    unittest.main()
