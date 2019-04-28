import unittest


import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../lib")
import routing_algorithms
import routing_simulation
import graph_edge_factory
import graph
from collections import deque


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
        """
        Testing
        (a) If the link is removed after the swap operation
        (b) If we get a waiting time of 1 for each neighbouring edge and no-link distance of 0
        """
        for dth in range(1, 5):
            threshold = 2 ** dth
            factory = graph_edge_factory.GraphEdgesFactory(distance_threshold=threshold, original_capacity=1)

            # Check for deterministic graph
            deterministic_edges = factory.generate_deterministic_graph_edges(factory.deterministic_link)
            main_graph = graph.Graph(deterministic_edges)

            # For each of the edge, we increment the waiting time, but not the number of no-links
            for (start, end, capacity) in deterministic_edges:
                for x in range(capacity):
                    self.assertEqual(main_graph.get_edge_capacity(start, end), capacity-x)
                    self.assertEqual(routing_algorithms.entanglement_swap(main_graph, start, end), (1, 0))

                    # The link was removed from the graph
                    self.assertEqual(main_graph.get_edge_capacity(start, end), capacity - x-1)


class TestDistributeEntanglement(unittest.TestCase):
    def test_on_demand_distribute_entanglement(self):
        factory = graph_edge_factory.GraphEdgesFactory(original_capacity=0)
        deterministic_edges = factory.generate_deterministic_graph_edges(factory.deterministic_link)
        main_graph = graph.Graph(deterministic_edges)
        for x in range(3, factory.number_of_nodes+1):
            path = [node for node in range(1, x)]
            local_settings = routing_simulation.Settings()
            unit_time_for_rebuild = (1/local_settings.rebuild_probability)

            potential_waiting_time = unit_time_for_rebuild**(len(path)-1)

            if local_settings.time_threshold > potential_waiting_time:
                self.assertEqual(routing_algorithms.distribute_entanglement(main_graph, path), potential_waiting_time)
            else:
                self.assertEqual(routing_algorithms.distribute_entanglement(main_graph, path),
                                 unit_time_for_rebuild**main_graph.dist(path[-1:][0], path[:1][0]))

    def test_continuous_distribute_entanglement(self):
        factory = graph_edge_factory.GraphEdgesFactory(distance_threshold=2, number_of_nodes=32, max_threshold=4)
        graph_edges = factory.generate_deterministic_graph_edges(factory.deterministic_link)
        local_graph = graph.Graph(graph_edges)

        # First consume links (which costs 1), then assert for the real cost 4 ** dist
        # 1-> 2
        self.assertEqual(1, routing_algorithms.distribute_entanglement(local_graph, [1, 2]))
        self.assertEqual(4, routing_algorithms.distribute_entanglement(local_graph, [1, 2]))

        # 2-> 3
        self.assertEqual(1, routing_algorithms.distribute_entanglement(local_graph, [2, 3]))
        self.assertEqual(4, routing_algorithms.distribute_entanglement(local_graph, [2, 3]))

        # 1-> 3
        for x in range(local_graph.get_edge_capacity(1, 3)):
            self.assertEqual(1, routing_algorithms.distribute_entanglement(local_graph, [1, 3]))

        self.assertEqual(4 ** local_graph.dist(1, 3), routing_algorithms.distribute_entanglement(local_graph, [1, 3]))

        # 1-> 5
        for x in range(local_graph.get_edge_capacity(3, 5)):
            self.assertEqual(4 ** local_graph.dist(1, 3) + 1,
                             routing_algorithms.distribute_entanglement(local_graph, [1, 3, 5]))

        self.assertEqual(4 ** local_graph.dist(1, 5),
                         routing_algorithms.distribute_entanglement(local_graph, [1, 3, 5]))




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
        factory = graph_edge_factory.GraphEdgesFactory(distance_threshold=4, max_threshold=4)
        graph_edges = factory.generate_deterministic_graph_edges(factory.deterministic_link)
        local_graph = graph.Graph(graph_edges)

        # self.assertEqual(routing_algorithms.dijkstra(local_graph, 1, 17), [17, 13, 9, 5, 1])
        self.assertEqual(routing_algorithms.dijkstra(local_graph, 1, 27), [27, 29, 1])
        self.assertEqual(routing_algorithms.dijkstra(local_graph, 1, 26), [26, 27, 29, 1])
        self.assertEqual(routing_algorithms.dijkstra(local_graph, 1, 25), [25, 29, 1])
        self.assertEqual(routing_algorithms.dijkstra(local_graph, 1, 24), [24, 25, 29, 1])
        self.assertEqual(routing_algorithms.dijkstra(local_graph, 1, 23), [23, 25, 29, 1])
        self.assertEqual(routing_algorithms.dijkstra(local_graph, 1, 22), [22, 23, 25, 29, 1])
        self.assertEqual(routing_algorithms.dijkstra(local_graph, 1, 21), [21, 25, 29, 1])
        self.assertEqual(routing_algorithms.dijkstra(local_graph, 1, 20), [20, 21, 25, 29, 1])
        self.assertEqual(routing_algorithms.dijkstra(local_graph, 1, 19), [19, 21, 25, 29, 1])
        self.assertEqual(routing_algorithms.dijkstra(local_graph, 1, 18), [18, 17, 13, 9, 5, 1])
        self.assertEqual(routing_algorithms.dijkstra(local_graph, 16, 1), [1, 29, 25, 21, 17, 16])
        self.assertEqual(routing_algorithms.dijkstra(local_graph, 1, 17), [17, 13, 9, 5, 1])
        self.assertEqual(routing_algorithms.dijkstra(local_graph, 1, 16), [16, 17, 13, 9, 5, 1])
        self.assertEqual(routing_algorithms.dijkstra(local_graph, 1, 15), [15, 13, 9, 5, 1])
        self.assertEqual(routing_algorithms.dijkstra(local_graph, 1, 14), [14, 13, 9, 5, 1])

    def test_dth8_long_links_path(self):
        """
        Test that it can sum a list of integers
        """
        factory = graph_edge_factory.GraphEdgesFactory(distance_threshold=8)
        graph_edges = factory.generate_deterministic_graph_edges(factory.deterministic_link)
        local_graph = graph.Graph(graph_edges)

        # self.assertEqual(routing_algorithms.dijkstra(local_graph, 1, 17), [17, 13, 9, 5, 1])
        self.assertEqual(routing_algorithms.dijkstra(local_graph, 1, 19), [19, 17, 9, 1])
        self.assertEqual(routing_algorithms.dijkstra(local_graph, 19, 1), [1, 9, 17, 19])
        self.assertEqual(routing_algorithms.dijkstra(local_graph, 1, 18), [18, 17, 9, 1])
        self.assertEqual(routing_algorithms.dijkstra(local_graph, 18, 1), [1, 9, 17, 18])
        self.assertEqual(routing_algorithms.dijkstra(local_graph, 1, 16), [16, 17, 9, 1])
        self.assertEqual(routing_algorithms.dijkstra(local_graph, 1, 15), [15, 17, 9, 1])
        self.assertEqual(routing_algorithms.dijkstra(local_graph, 1, 14), [14, 13, 9, 1])

    def test_dth16_long_links_path(self):
        """
        Test that it can sum a list of integers
        """
        factory = graph_edge_factory.GraphEdgesFactory(distance_threshold=16)
        graph_edges = factory.generate_deterministic_graph_edges(factory.deterministic_link)
        local_graph = graph.Graph(graph_edges)

        self.assertEqual(routing_algorithms.dijkstra(local_graph, 1, 17), [17, 1])

        self.assertTrue(routing_algorithms.dijkstra(local_graph, 6, 18) != [18, 17, 1, 5, 6])

        self.assertEqual(routing_algorithms.dijkstra(local_graph, 32, 18), [18, 17, 1, 32])
        self.assertEqual(routing_algorithms.dijkstra(local_graph, 31, 18), [18, 17, 1, 31])
        self.assertEqual(routing_algorithms.dijkstra(local_graph, 31, 18), [18, 17, 1, 31])
        self.assertEqual(routing_algorithms.dijkstra(local_graph, 30, 18), [18, 17, 1, 29, 30])
        self.assertEqual(routing_algorithms.dijkstra(local_graph, 29, 18), [18, 17, 1, 29])
        self.assertEqual(routing_algorithms.dijkstra(local_graph, 28, 18), [18, 17, 1, 29, 28])

        self.assertTrue(routing_algorithms.dijkstra(local_graph, 27, 18) != [18, 17, 1, 29, 27])
        self.assertTrue(routing_algorithms.dijkstra(local_graph, 6, 18) != [18, 17, 1, 5, 6])

    def test_dth4_on_demand_long_links_path(self):
        """
        Test that it can sum a list of integers
        """
        factory = graph_edge_factory.GraphEdgesFactory(distance_threshold=4, original_capacity=0)
        graph_edges = factory.generate_deterministic_graph_edges(factory.deterministic_link)
        local_graph = graph.Graph(graph_edges)

        # self.assertEqual(routing_algorithms.dijkstra(local_graph, 1, 17), [17, 13, 9, 5, 1])
        self.assertEqual(routing_algorithms.dijkstra(local_graph, 1, 16), [16, 15, 13, 9, 5, 1])
        self.assertEqual(routing_algorithms.dijkstra(local_graph, 16, 1), [1, 5, 9, 13, 15, 16])
        self.assertEqual(routing_algorithms.dijkstra(local_graph, 1, 15), [15, 13, 9, 5, 1])
        self.assertEqual(routing_algorithms.dijkstra(local_graph, 15, 1), [1, 5, 9, 13, 15])
        self.assertEqual(routing_algorithms.dijkstra(local_graph, 1, 14), [14, 13, 9, 5, 1])
        self.assertEqual(routing_algorithms.dijkstra(local_graph, 14, 1), [1, 5, 9, 13, 14])


    def test_dth8_on_demand_long_links_path(self):
        """
        Test that it can sum a list of integers
        """
        factory = graph_edge_factory.GraphEdgesFactory(distance_threshold=8, original_capacity=0)
        graph_edges = factory.generate_deterministic_graph_edges(factory.deterministic_link)
        local_graph = graph.Graph(graph_edges)

        # self.assertEqual(routing_algorithms.dijkstra(local_graph, 1, 17), [17, 13, 9, 5, 1])
        self.assertEqual(routing_algorithms.dijkstra(local_graph, 1, 19), [19, 21, 25, 1])

        # Disclaimer: the following is longer than the expected shortest path, as we are in the on-demand model
        self.assertEqual(routing_algorithms.dijkstra(local_graph, 1, 18), [18, 19, 21, 25, 1])
        self.assertEqual(routing_algorithms.dijkstra(local_graph, 1, 16), [16, 15, 13, 9, 1])
        self.assertEqual(routing_algorithms.dijkstra(local_graph, 1, 15), [15, 13, 9, 1])
        self.assertEqual(routing_algorithms.dijkstra(local_graph, 1, 14), [14, 13, 9, 1])


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


class TestLinkPredictionDijkstra(unittest.TestCase):
    def test_dth2_simple_path(self):
        """
        Test that it can sum a list of integers
        """
        factory = graph_edge_factory.GraphEdgesFactory(distance_threshold=2)
        graph_edges = factory.generate_deterministic_graph_edges(factory.deterministic_link)
        local_graph = graph.Graph(graph_edges)

        self.assertEqual(routing_algorithms.dijkstra(local_graph, 1, 3), [3, 1], False)


class TestEntanglement(unittest.TestCase):
    def test_entanglement_swap(self):
        routing_algorithms.entanglement_swap()


'''
Functions called upon in the local knowledge method: graph.Graph(), graph.available_virtual_link_count(),
                                                                    graph.available_edge_count()
                                                    initialize_paths: gen_rand_pairs, dijkstra,
                                                    get_simulation_data_for_paths: distribute_entanglement,
                                                    entanglement_swap
                                                    
'''


class TestInitialKnowledge(unittest.TestCase):
    def test_gen_rand_pairs(self):

        number_of_pairs = 100
        generated_pairs = routing_algorithms.gen_rand_pairs(number_of_pairs)
        number_of_nodes = routing_simulation.Settings().number_of_nodes

        self.assertEqual(number_of_pairs, len(generated_pairs))
        [self.assertTrue(x[0] != x[1]) for x in generated_pairs]
        [self.assertTrue(0 < x[0] < number_of_nodes + 1 and 0 < x[1] < number_of_nodes + 1) for x in generated_pairs]

    def test_initialize_paths(self):
        number_of_pairs = 100
        factory = graph_edge_factory.GraphEdgesFactory(distance_threshold=2)
        graph_edges = factory.generate_deterministic_graph_edges(factory.deterministic_link)
        local_graph = graph.Graph(graph_edges)
        self.assertTrue(number_of_pairs, len(routing_algorithms.initialize_paths(local_graph, number_of_pairs)))

    def test_serve_demands(self):

        factory = graph_edge_factory.GraphEdgesFactory(distance_threshold=2, number_of_nodes=32, max_threshold=4)
        graph_edges = factory.generate_deterministic_graph_edges(factory.deterministic_link)
        local_graph = graph.Graph(graph_edges)

        results = routing_algorithms.serve_demands(local_graph, deque([(1, 2)]))
        self.assertEqual(results, ([1], [55], [47]))


# TODO implement tests for these methods
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
