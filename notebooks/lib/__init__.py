import time
import routing_simulation
import routing_algorithms
import graph_edge_factory
import plot
import logger

if __name__ == "__main__":


    initial_knowledge_results = []
    samples = 1
    max_th = 3
    start = time.time()
    det_graphs = []
    for dth in range(4, 5):
        threshold = 2 ** dth
        factory = graph_edge_factory.GraphEdgesFactory(distance_threshold=threshold, max_threshold=16)
        graph_edges = factory.generate_deterministic_graph_edges(factory.deterministic_link)
        arguments = {'algorithm': routing_algorithms.initial_knowledge_init, 'graph_edges': graph_edges,
                     'link_prediction': False, 'exponential_scale': True}
        local_result, length = routing_simulation.run_algorithm_for_graphs(50, samples, arguments)
        initial_knowledge_results.append(local_result)
        end = time.time()
        print(det_graphs)
    plot.plot_results(initial_knowledge_results, 'global_knowledge_dth_' + str(threshold) + '_maxdth_' + str(max_th),
                      save_tikz = False)