import time
import routing_simulation
import routing_algorithms
import graph_edge_factory
import plot

if __name__ == "__main__":

    # Example code for running an initial knowledge simulation in a graph of 32 nodes.
    # We specify a varying value for the distance threshold (dth), whereas the maximum distance threshold (max dth)
    # is constant (additional edges are added for the case when the dth is not equal to max dth
    global_knowledge_results = []
    samples = 10
    max_dth = 16
    start = time.time()
    for dth in range(2, 4):
        threshold = 2 ** dth
        factory = graph_edge_factory.VirtualEdgeFactory(distance_threshold=threshold, max_distance_threshold=max_dth)
        graph_edges = factory.generate_deterministic_graph_edges()
        print(graph_edges)
        arguments = {'algorithm': routing_algorithms.initial_knowledge_init, 'graph_edges': graph_edges,
                     'link_prediction': False, 'exponential_scale': True}
        topology_result, length = routing_simulation.run_algorithm_for_graphs(50, samples, arguments)
        global_knowledge_results.append(topology_result)
    end = time.time()
    plot.plot_results(global_knowledge_results, 'global_knowledge_maxdth_' + str(max_dth) + str(end-start),
                      save_tikz = False)