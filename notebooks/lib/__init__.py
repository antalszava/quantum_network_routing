import time
import routing_simulation
import routing_algorithms
import graph_edge_factory

if __name__ == "__main__":

    global_knowledge_results = []
    start = time.time()
    samples = 10
    for dth in range(1, 5):
        threshold = 2 ** 2
        factory = graph_edge_factory.GraphEdgesFactory(distance_threshold=threshold)
        graph_edges = factory.generate_random_power_law_graph_edges()
        arguments = {'algorithm': routing_algorithms.global_knowledge_init, 'graph_edges': graph_edges}
        result, length = routing_simulation.run_algorithm_for_graphs(50, samples, arguments)
        global_knowledge_results.append(result)
    end = time.time()
