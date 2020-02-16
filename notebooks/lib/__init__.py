import time
import routing_simulation
import routing_algorithms
import graph_edge_factory
import helper
import plot
import numpy as np

if __name__ == "__main__":

    # Example code for running an initial knowledge simulation in a graph of 32 nodes.
    # We specify a varying value for the distance threshold (dth), whereas the maximum distance threshold (max dth)
    # is constant (additional edges are added for the case when the dth is not equal to max dth

    samples = 5
    threshold = 4
    max_dth = 4
    number_of_nodes = 32
    link_prediction_betweenness = routing_simulation.LinkPredictionTypes.Betweenness
    link_prediction_iterative_betweenness = routing_simulation.LinkPredictionTypes.IterativeBetweenness
    link_prediction_closeness = routing_simulation.LinkPredictionTypes.Closeness

    initial_knowledge_settings = routing_simulation.AlgorithmSettings(algorithm=routing_algorithms.initial_knowledge_init)
    initial_knowledge_betweennes_settings = routing_simulation.AlgorithmSettings(algorithm=
                                                                            routing_algorithms.initial_knowledge_init,
                                                                            link_prediction=link_prediction_betweenness)

    initial_knowledge_iterative_betweennes_settings = routing_simulation.AlgorithmSettings(algorithm=
                                                                                 routing_algorithms.initial_knowledge_init,
                                                                                 link_prediction=
                                                                                 link_prediction_iterative_betweenness)

    initial_knowledge_closenness_settings = routing_simulation.AlgorithmSettings(algorithm=
                                                                                 routing_algorithms.initial_knowledge_init,
                                                                                 link_prediction=link_prediction_closeness)

    algorithm_settings = [initial_knowledge_settings, initial_knowledge_betweennes_settings]
    simulation_settings = routing_simulation.SimulationSettings(number_of_samples=samples)

    factory = graph_edge_factory.VirtualEdgeFactory(number_of_nodes=number_of_nodes,
                                                    distance_threshold=threshold, max_distance_threshold=max_dth)
    graph_edges = factory.generate_deterministic_graph_edges()

    topology_settings = routing_simulation.TopologySettings(number_of_nodes=number_of_nodes,
                                                            graph_edges=graph_edges, distance_threshold=threshold)

    simulation = routing_simulation.Simulation(simulation_settings, topology_settings, algorithm_settings)

    start = time.time()
    simulation.run_algorithm_for_graphs()
    end = time.time()
    print(end-start)
    plot.plot_results(list(simulation.final_results.values()), '_maxdth_'
                          + str(max_dth) + str(end - start), save_tikz=False)

    '''
    # On-demand
    factory = graph_edge_factory.VirtualEdgeFactory(distance_threshold=threshold, max_distance_threshold=max_dth)
    graph_edges = factory.generate_deterministic_graph_edges()
    arguments = {'algorithm': routing_algorithms.initial_knowledge_init, 'graph_edges': graph_edges,
                 'link_prediction': False, 'exponential_scale': True}
    topology_result, errors, length = routing_simulation.run_algorithm_for_graphs(50, samples, arguments)
    initial_knowledge_results.append(topology_result)
    plot.plot_results(initial_knowledge_results, 'initial_knowledge_maxdth_' + str(max_dth) + str(end - start),
                 save_tikz=False)
    initial_knowledge_results = []
    initial_knowledge_errors = []
    samples = 2
    max_dth = 16
    start = time.time()
    for dth in range(1, 5):
        threshold = 2 ** dth
        power_law_results = []
        for sampling_power_law in range(10):
            factory = graph_edge_factory.VirtualEdgeFactory(distance_threshold=threshold, max_distance_threshold=max_dth)
            graph_edges = factory.generate_random_power_law_graph_edges()
            arguments = {'algorithm': routing_algorithms.initial_knowledge_init, 'graph_edges': graph_edges,
                         'link_prediction': False, 'exponential_scale': True}
            local_result, errors, length = routing_simulation.run_algorithm_for_graphs(50, samples, arguments)
            power_law_results.append(local_result)
        temp = [list(zip(*item)) for item in list(zip(*power_law_results))]
        topology_result = [helper.map_tuple_gen(np.mean, x) for x in temp]
        initial_knowledge_results.append(topology_result)
        initial_knowledge_errors.append(errors)
    end = time.time()
    plot.plot_results(initial_knowledge_results, 'initial_knowledge_random_k1_graph_dth_' + str(max_dth) + str(end - start),
                 save_tikz=False)
    '''
