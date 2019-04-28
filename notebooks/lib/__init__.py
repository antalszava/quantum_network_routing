import time
import graph
import routing_simulation
import routing_algorithms
import graph_edge_factory
import plot
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import write_dot

if __name__ == "__main__":
    '''
    factory = graph_edge_factory.GraphEdgesFactory(distance_threshold = 4)
    graph1 = factory.generate_random_power_law_graph_edges()

    threshold = 2 ** 4
    factory = graph_edge_factory.GraphEdgesFactory(distance_threshold=threshold)
    graph_edges = factory.generate_random_power_law_graph_edges()

    print(graph_edges)

    G = nx.MultiGraph()
    G.add_edges_from(graph_edges)
    nx.draw_circular(G, with_labels=True)

    plt.savefig('/home/antal/Documents/eit/thesis/implementation/quantum_routing/notebooks/plots/random_power_law/'
                'graph_images/' + 'some_graph.png', bbox_inches='tight')
    plt.show()
    
    G.add_edges_from(

            # Iniating the edges of 0) type: 1->2, 2->3, ... 31->32, 32->1 (circular graph)
            [(x, x + 1, {'capacity': original_capacity}) for x in range(1, number_of_nodes)] + [(number_of_nodes, 1)]
        )
    '''

    '''
    initial_knowledge_results = []

    for dth in range(0, 3):
        threshold = 2 ** dth
        factory = graph_edge_factory.GraphEdgesFactory(distance_threshold=threshold, max_threshold =4)
        graph_edges = factory.generate_deterministic_graph_edges(factory.deterministic_link)

        print(graph_edges)

        arguments = {'algorithm': routing_algorithms.initial_knowledge_init, 'graph_edges': graph_edges}
        results, length = routing_simulation.run_algorithm_for_graphs(75, 100, arguments)
        initial_knowledge_results.append(results)
        '''
    initial_knowledge_results = []
    #for dth in range(0, 3):
    threshold = 2 ** 2
    factory = graph_edge_factory.GraphEdgesFactory(distance_threshold=threshold, max_threshold=4)
    graph_edges = factory.generate_deterministic_graph_edges(factory.deterministic_link)
    arguments = {'algorithm': routing_algorithms.initial_knowledge_init, 'graph_edges': graph_edges}
    result, length = routing_simulation.run_algorithm_for_graphs(50, 1, arguments)
    initial_knowledge_results.append(result)

    print(initial_knowledge_results)
    '''
    G = nx.MultiGraph()

    factory = graph_edge_factory.GraphEdgesFactory(number_of_nodes=16, distance_threshold=4)
    graph_edges = factory.generate_deterministic_graph_edges(factory.deterministic_link)

    G.add_edges_from(graph_edges)
    #nx.draw_circular(G, with_labels=True, edge_color='r', style='dashed', font_size='14',
    #                 font_weight='bold', node_color='w', width=3, )
    print(G.edges)

    # labels = nx.get_edge_attributes(G,'weight')
    # nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)

    temp_graph = graph.Graph(graph_edges)
    current_frequencies = routing_algorithms.get_edge_frequencies_in_graph(temp_graph)

    for edge in graph_edges:
        start = edge[0] if edge[0] < edge[1] else edge[1]
        end = edge[1] if edge[0] > edge[1] else edge[0]
        G.edges[edge]['weight'] = current_frequencies[(start, end)]

    
    for edge_freq in current_frequencies:
        G.edges[edge_freq]['weight'] = 1  # current_frequencies[edge_freq]
    
    print(current_frequencies)

    # plt.savefig('/home/antal/Documents/eit/thesis/talks/thesis_summary/' + str(2) + '.png', bbox_inches='tight')
    # plt.clf()


    #plot.plot_results(initial_knowledge_results, 'initial_knowledge')
    

    # global_knowledge_results = []

    for i in range(1, 3):
        arguments = {'algorithm': routing_algorithms.global_algo, 'distance_threshold': 2 ** i}
        start = time.time()
        global_knowledge_results.append(routing_simulation.run_algorithm_for_graphs(routing_simulation.Settings().
                                                                                    number_of_source_destination_pairs,
                                                                                    routing_simulation.Settings().
                                                                                    number_of_samples,
                                                                                    arguments))
    end = time.time()

    print(global_knowledge_results)

    initial_knowledge_results = []

    for i in range(1, 3):
        arguments = {'algorithm': routing_algorithms.initial_knowledge_init, 'distance_threshold': 2 ** i}
        start = time.time()
        initial_knowledge_results.append(routing_simulation.run_algorithm_for_graphs(routing_simulation.Settings().
                                                                                     number_of_source_destination_pairs,
                                                                  routing_simulation.Settings().number_of_samples,
                                                                  arguments))
    end = time.time()


    global_knowledge_results = []
    for i in range(0, 3):
        arguments = {'algorithm': routing_algorithms.global_algo, 'distance_threshold': 2 ** i}
        global_knowledge_results.append(routing_simulation.run_algorithm_for_graphs(routing_simulation.Settings().
                                                                                    number_of_source_destination_pairs,
                                                                                    routing_simulation.Settings().number_of_samples,
                                                                                    arguments))
    print(global_knowledge_results)

    global_knowledge_results = []

    for i in range(1, 3):
        arguments = {'algorithm': routing_algorithms.global_algo, 'distance_threshold': 2 ** i}
        start = time.time()
        global_knowledge_results.append(routing_simulation.run_algorithm_for_graphs(routing_simulation.Settings().
                                                                                    number_of_source_destination_pairs,
                                                                                    routing_simulation.Settings().number_of_samples,
                                                                                    arguments))
    end = time.time()
    print(global_knowledge_results)
    
    for radius in range(0, 11):

        local_knowledge = []
        for i in range(1, 3):
            arguments = {'algorithm': routing_algorithms.local_knowledge_algorithm,
                         'distance_threshold': 2 ** i, 'knowledge_radius': radius}

            start = time.time()
       
            local_knowledge += [routing_simulation.run_algorithm_for_graphs(routing_simulation.Settings().
                                                         number_of_source_destination_pairs,
                                                         routing_simulation.Settings().number_of_samples, arguments)]
            

            global_knowledge_results = [routing_simulation.run_algorithm_for_graphs(routing_simulation.Settings().
                                                                                    number_of_source_destination_pairs,
                                        routing_simulation.Settings().number_of_samples, arguments)]
            end = time.time()
            print(global_knowledge_results)

    factory = graph_edge_factory.GraphEdgesFactory()
    graph1 = factory.generate_deterministic_graph_edges(factory.deterministic_link)

    factory = graph_edge_factory.GraphEdgesFactory(distance_threshold=2)
    graph2 = factory.generate_deterministic_graph_edges(factory.deterministic_link)

    factory = graph_edge_factory.GraphEdgesFactory(distance_threshold=4)
    graph3 = factory.generate_deterministic_graph_edges(factory.deterministic_link)

    det_graphs = [
    [(1, 2, 0), (2, 3, 0), (3, 4, 0), (4, 5, 0), (5, 6, 0), (6, 7, 0), (7, 8, 0), (8, 9, 0), (9, 10, 0),
     (10, 11, 0), (11, 12, 0), (12, 13, 0), (13, 14, 0), (14, 15, 0), (15, 16, 0), (16, 17, 0), (17, 18, 0),
     (18, 19, 0), (19, 20, 0), (20, 21, 0), (21, 22, 0), (22, 23, 0), (23, 24, 0), (24, 25, 0), (25, 26, 0),
     (26, 27, 0), (27, 28, 0), (28, 29, 0), (29, 30, 0), (30, 31, 0), (31, 32, 0), (32, 1, 0)], graph1, graph2,
    graph3]
    print(det_graphs)

    my_graph = graph.Graph(graph1)

    arguments = {'algorithm': routing_algorithms.global_algo, 'graphs': det_graphs}
    start = time.time()
    global_knowledge_results = [routing_simulation.run_algorithm_for_graphs(routing_simulation.Settings().
                                                                             number_of_source_destination_pairs,
                                                          routing_simulation.Settings().number_of_samples, arguments)]

    print(global_knowledge_results)

    routing_algorithms.dijkstra(my_graph, 3, 24)

'''