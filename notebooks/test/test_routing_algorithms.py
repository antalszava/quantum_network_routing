import unittest


import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../lib")
import routing_algorithms
import routing_simulation
import graph_edge_factory
import graph
import shortest_path
import math
import networkx as nx
from collections import deque


class TestTracebackPath(unittest.TestCase):
    def test_list_int(self):
        """
        Test that it can sum a list of integers
        """
        data = {3: None, 4: 3, 7: 4, 12: 16, 16: 7, 2: 12}
        result = shortest_path.traceback_path(2, data)
        self.assertEqual(result, [2, 12, 16, 7, 4, 3])

    def test_end_of_path_not_destination(self):
        """
        Test that when tracing back the path, then we don't end up at the destination (that would be 16)
        """
        data = {16: None, 7: 3, 12: 7, 2: 12}
        self.assertRaises(KeyError, shortest_path.traceback_path, 2, data)


class TestEntanglementSwap(unittest.TestCase):
    def test_on_demand_graph(self):

        factory = graph_edge_factory.VirtualEdgeFactory(distance_threshold=1, capacity=0)
        graph_edges = factory.generate_deterministic_graph_edges()
        main_graph = graph.Graph(graph_edges)

        # For each of the edge, we don't increase the latency, but increment the number of no-links
        for (start, end, capacity) in graph_edges:
            self.assertEqual(capacity, 0)
            self.assertEqual(routing_algorithms.entanglement_swap(main_graph, start, end), (0, 1))

    def test_dth1(self):
        """
        Testing
        (a) If the link is removed after the swap operation
        (b) If we get a latency of 1 for each neighbouring edge and no-link distance of 0
        """
        for dth in range(1, 5):
            threshold = 2 ** dth
            factory = graph_edge_factory.VirtualEdgeFactory(distance_threshold=threshold, capacity=1)

            # Check for deterministic graph
            deterministic_edges = factory.generate_deterministic_graph_edges()
            main_graph = graph.Graph(deterministic_edges)

            # For each of the edge, we increment the latency, but not the number of no-links
            for (start, end, capacity) in deterministic_edges:
                for x in range(capacity):
                    self.assertEqual(main_graph.get_edge_capacity(start, end), capacity-x)
                    self.assertEqual(routing_algorithms.entanglement_swap(main_graph, start, end), (1, 0))

                    # The link was removed from the graph
                    self.assertEqual(main_graph.get_edge_capacity(start, end), capacity - x-1)


class TestDistributeEntanglement(unittest.TestCase):
    def test_on_demand_distribute_entanglement(self):
        factory = graph_edge_factory.VirtualEdgeFactory(capacity=0)
        deterministic_edges = factory.generate_deterministic_graph_edges()
        main_graph = graph.Graph(deterministic_edges)
        for x in range(3, factory.number_of_nodes+1):
            path = [node for node in range(1, x)]
            local_settings = routing_simulation.Settings()
            unit_time_for_rebuild = (1/local_settings.rebuild_probability)

            potential_latency = unit_time_for_rebuild**(len(path)-1)

            if local_settings.time_threshold > potential_latency:
                self.assertEqual(routing_algorithms.distribute_entanglement(main_graph, path), potential_latency)
            else:
                self.assertEqual(routing_algorithms.distribute_entanglement(main_graph, path),
                                 unit_time_for_rebuild ** main_graph.physical_distance(path[-1:][0], path[:1][0]))

    def test_continuous_distribute_entanglement(self):
        factory = graph_edge_factory.VirtualEdgeFactory(distance_threshold=2, number_of_nodes=32,
                                                        max_distance_threshold=4)
        graph_edges = factory.generate_deterministic_graph_edges()
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

        self.assertEqual(4 ** local_graph.physical_distance(1, 3), routing_algorithms.distribute_entanglement(local_graph, [1, 3]))

        # 1-> 5
        for x in range(local_graph.get_edge_capacity(3, 5)):
            self.assertEqual(4 ** local_graph.physical_distance(1, 3) + 1,
                             routing_algorithms.distribute_entanglement(local_graph, [1, 3, 5]))

        self.assertEqual(4 ** local_graph.physical_distance(1, 5),
                         routing_algorithms.distribute_entanglement(local_graph, [1, 3, 5]))



'''
Functions called upon in the local knowledge method: graph.Graph(), graph.available_virtual_link_count(),
                                                                    graph.available_edge_count()
                                                    initialize_paths: generate_random_pairs, dijkstra,
                                                    get_simulation_data_for_paths: distribute_entanglement,
                                                    entanglement_swap
                                                    
'''


class TestInitialKnowledge(unittest.TestCase):
    def test_generate_random_pairs(self):

        number_of_pairs = 100
        generated_pairs = routing_algorithms.generate_random_pairs(number_of_pairs)
        number_of_nodes = routing_simulation.Settings().number_of_nodes

        self.assertEqual(number_of_pairs, len(generated_pairs))
        [self.assertTrue(x[0] != x[1]) for x in generated_pairs]
        [self.assertTrue(0 < x[0] < number_of_nodes + 1 and 0 < x[1] < number_of_nodes + 1) for x in generated_pairs]

    def test_initialize_paths(self):
        number_of_pairs = 100
        factory = graph_edge_factory.VirtualEdgeFactory(distance_threshold=2)
        graph_edges = factory.generate_deterministic_graph_edges()
        local_graph = graph.Graph(graph_edges)
        self.assertTrue(number_of_pairs, len(routing_algorithms.initialize_paths(local_graph, number_of_pairs)))

    def test_serve_demands(self):

        factory = graph_edge_factory.VirtualEdgeFactory(distance_threshold=2, number_of_nodes=32,
                                                        max_distance_threshold=4)
        graph_edges = factory.generate_deterministic_graph_edges()
        local_graph = graph.Graph(graph_edges)

        results = routing_algorithms.serve_demands(local_graph, deque([(1, 2)]))
        self.assertEqual(results, ([1], [55], [47]))

    def test_initial_knowledge_init(self):
        factory = graph_edge_factory.VirtualEdgeFactory(distance_threshold=2, number_of_nodes=32,
                                                        max_distance_threshold=4)
        graph_edges = factory.generate_deterministic_graph_edges()
        results = routing_algorithms.initial_knowledge_init(graph_edges, 50, link_prediction=False)
        self.assertEqual(4, len(results))

    def test_initial_knowledge_step(self):
        factory = graph_edge_factory.VirtualEdgeFactory(distance_threshold=2, number_of_nodes=32,
                                                        max_distance_threshold=4)
        graph_edges = factory.generate_deterministic_graph_edges()
        local_graph = graph.Graph(graph_edges, link_prediction=True)

        local_final_results = ([], [], [], [])
        local_time_window_size = 5
        local_number_of_source_destination_pairs = 100

        for x in range(1, local_number_of_source_destination_pairs + 1):
            self.assertEqual(None, routing_algorithms.initial_knowledge_step(local_graph, x, local_time_window_size,
                                                                             local_number_of_source_destination_pairs,
                                                                             local_final_results))

        # See if we really have all the rounds computed
        [self.assertEqual(math.ceil(local_number_of_source_destination_pairs/local_time_window_size) *
                          local_time_window_size,
                          len(x)) for x in local_final_results]



# TODO implement tests for these methods
'''
Functions called upon in the local knowledge method: create_graph_with_local_knowledge, dijkstra,
                                                    distribute_entanglement, update_local_knowledge 
'''


class TestLocalKnowledge(unittest.TestCase):

    @staticmethod
    def does_node_have_correct_knowledge(local_knowledge, main_graph):
        does_node_have_correct_knowledge = True
        for vertex in local_knowledge.Vertices:
            does_node_have_correct_knowledge &= vertex in main_graph.Vertices
            for neighbour in local_knowledge.Vertices[vertex].neighbours:
                neighbour_properties = local_knowledge.Vertices[vertex].neighbours[neighbour]
                main_graph_neighbour = main_graph.Vertices[vertex].neighbours[neighbour]
                neighbour_properties.capacity = main_graph_neighbour.capacity
                neighbour_properties.max_capacity = main_graph_neighbour.max_capacity
                neighbour_properties.weight = main_graph_neighbour.weight
        return does_node_have_correct_knowledge

    def test_create_graph_(self):
        for dth in range(1, 5):
            threshold = 2 ** dth
            factory = graph_edge_factory.VirtualEdgeFactory(distance_threshold=threshold)

            # Check for deterministic graph
            deterministic_edges = factory.generate_deterministic_graph_edges()
            local_knowledge_graph = routing_algorithms.create_graph_with_local_knowledge(deterministic_edges)

            main_graph = graph.Graph(deterministic_edges)
            for vertex in local_knowledge_graph.Vertices.values():
                self.assertTrue(TestLocalKnowledge.does_node_have_correct_knowledge(vertex.local_knowledge, main_graph))
                self.assertTrue(TestLocalKnowledge.does_node_have_correct_knowledge(main_graph, vertex.local_knowledge))


if __name__ == '__main__':
    unittest.main()
