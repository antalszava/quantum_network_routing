import random
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../lib")

import routing_simulation
import helper
import graph
import shortest_path
import networkx as nx

from collections import deque
import numpy as np


def entanglement_swap(graph, start_node: int, end_node: int) -> tuple:
    """
    Processes the virtual link between a start and end node by checking the capacity of the link between them.

    If the capacity is 0, then a probabilistic rebuild approach is used.
    Else, the link is removed and the latency is incremented.

    Parameters
    ----------
    graph: Graph
        The graph in which we want to perform entanglement swap.
    start_node: int
        Index of the starting vertex, from which we are looking for the shortest path.
    end_node: int
        Index of the end vertex, towards which we are looking for the shortest path.

    Returns
    -----
        A tuple of the latency and the length of the path along the physical graph used for rebuilding
        The latter is equal to the physical distance between the starting node and the end node.

    Notes
    -----
        The latency coming from the rebuilding of the unavailable links is not computed by this function.

    """
    local_latency = 0
    local_rebuild_length = 0

    if graph.get_edge_capacity(start_node, end_node) == 0:

        local_rebuild_length = graph.physical_distance(start_node, end_node)
    else:

        # Remove the link between startNode and endNode
        graph.remove_virtual_link(start_node, end_node)

        # Incrementing the latency
        local_latency = 1

    return local_latency, local_rebuild_length


def compute_latency_to_rebuild(graph, initial_node: int, end_node: int,
                               no_link_length: int, exponential_scale: bool = True) -> tuple:
    """
    Processes the virtual links used in the current path and sums the latency

    Parameters
    ----------
    graph: Graph
        The graph in which we run our simulation.
    start_node: int
        Index of the starting vertex, from which we are looking for the shortest path.
    end_node: int
        Index of the end vertex, towards which we are looking for the shortest path.

    no_link_length: int
        The overall length of the virtual links that need to be rebuilt.

    exponential_scale: bool
        Value determining whether or not the latency scales exponentially with the distance.
        If the value if False, a polynomial scaling is used.

    Returns
    -----
        A tuple of latency and a value of whether or not the rebuild could take place within the time window.

    Notes
    -----
        If the rebuilding process cannot take place within a pre-defined threshold time value, then we simply start
        rebuilding along the physical graph and compute the latency accordingly.
    """
    latency_to_rebuild = 0
    could_rebuild_in_time_window = True

    # Check if there are links which were not available through the path
    if no_link_length > 0:

        # If we cannot create the missing entangled links in the specific threshold time
        # Then simply generate entangled links along the physical graph
        local_settings = routing_simulation.Settings()
        successful_rebuild_time = 1 / local_settings.rebuild_probability

        time_to_rebuild_path = successful_rebuild_time ** no_link_length \
            if exponential_scale else no_link_length ** 2

        if local_settings.time_threshold < time_to_rebuild_path:

            could_rebuild_in_time_window = False

            if exponential_scale:
                latency_to_rebuild = successful_rebuild_time ** graph.physical_distance(initial_node, end_node)
            else:
                latency_to_rebuild = graph.physical_distance(initial_node, end_node) ** 2
        else:
            latency_to_rebuild += time_to_rebuild_path

    return latency_to_rebuild, could_rebuild_in_time_window


def distribute_entanglement(graph, current_path: list, exponential_scale: bool = True):
    """
    Processes the virtual links used in the current path and sums the latency

    Parameters
    ----------
    graph: Graph
        The graph in which we run our simulation.
    current_path: list
        Index of the starting vertex, from which we are looking for the shortest path.
    exponential_scale: bool
        Value determining whether or not the latency scales exponentially with the distance.
        If the value if False, a polynomial scaling is used.

    Returns
    -----
        The latency in time steps between starting the procedure and using the path.

    Notes
    -----
        We count using an available virtual link as one time step, whereas the latency for rebuilding the unavailable
        virtual links is computed based on how the latency scales with the length of the path to be rebuilt
        (exponential or polynomial)
    """

    # Initializing entanglement delay time
    current_latency = 0
    no_link_length = 0
    remainder_of_path = deque(current_path)
    initial_node = remainder_of_path.popleft()
    get_initial = True

    # Take the leftmost two nodes out of the deque and get the latency until we are finished

    while True:
        if get_initial:
            start_node = initial_node
            get_initial = False
        else:
            start_node = remainder_of_path.popleft()
        end_node = remainder_of_path.popleft()

        # Calculate the latency and the no link length for the current step
        step_latency, step_no_link_length = entanglement_swap(graph, start_node, end_node)
        current_latency += step_latency
        no_link_length += step_no_link_length

        # Check if we have processed the path
        if len(remainder_of_path) == 0:

            # Rebuild the missing virtual links
            latency_to_rebuild, could_rebuild_in_time_window =\
                compute_latency_to_rebuild(graph, initial_node, end_node, no_link_length, exponential_scale)

            if could_rebuild_in_time_window:
                overall_latency = current_latency + latency_to_rebuild
            else:
                overall_latency = latency_to_rebuild

            return overall_latency

        # If there are still nodes to be processed, the end node is put back into the deque
        remainder_of_path.appendleft(end_node)


def serve_demands(graph, paths: deque, exponential_scale: bool = True) -> tuple:
    """
    Serving demands in the quantum network specified by paths and computing the latency for each.
    Also keeping track the remaining virtual links and overall virtual link capacity in the graph.

    Parameters
    ----------
    graph: Graph
        The graph in which we run our simulation.
    paths: deque
        The paths
    exponential_scale: bool
        Value determining whether or not the latency scales exponentially with the distance.
        If the value if False, a polynomial scaling is used.

    Returns
    -----
        A tuple of three lists, for each path the lists contain the following elements:
        -Latency
        -Sum of the available capacities in the virtual graph
        -Number of available virtual links

    """
    latency_store = []
    capacities_store = []
    virtual_link_store = []

    while True:

        current_path = paths.popleft()
        latency_store.append(distribute_entanglement(graph, current_path, exponential_scale))
        capacities_store.append(graph.get_sum_of_link_capacities())
        virtual_link_store.append(graph.get_available_link_count())

        if len(paths) == 0:
            return latency_store, capacities_store, virtual_link_store


def initialize_paths(graph, source_destination_pairs: list, link_prediction: bool = False) -> deque:
    """
    Initialise paths by generating source and destination pairs and finding the shortest path for each of them.

    Parameters
    ----------
    graph: Graph
        The graph in which we run our simulation.

    source_destination_pairs: list
        The source and destination pairs for which paths are to be initliased.

    link_prediction: bool
        Value determining whether or not link prediction is used.

    Returns
    -----
        A deque of shortest paths
    """

    # Assemble paths into one deque
    paths = deque()
    for pair in source_destination_pairs:
        # all_shortest_paths = list(nx.all_shortest_paths(graph.G, pair[0], pair[1], weight='weight'))

        if not link_prediction:
            path = shortest_path.dijkstra(graph, pair[0], pair[1])
        else:
            path = shortest_path.dijkstra(graph, pair[0], pair[1], link_prediction=link_prediction)

        paths.appendleft(path)

        '''
        np.random.seed()
        index = np.random.choice(range(len(all_shortest_paths)))
        path = all_shortest_paths[index]
        '''
    return paths


def create_graph_with_local_knowledge(graph_edges: list):
    """
    A helper method to create a graph with local knowledge.

    Parameters
    ----------
    graph_edges: list
        The graph edges to be added as local knowledge to the vertices of the graph.

    Returns
    -----
        A graph with local knowledge for each vertex.
    """
    temp_graph = graph.Graph(graph_edges)
    temp_graph.add_local_knowledge(graph_edges)
    return temp_graph


def update_along_physical_graph(main_graph, start_node: int, end_node: int, current_path: list):
    """
    Update the knowledge of nodes along the physical graph about the virtual links to be used in the current path.

    Parameters
    ----------
    main_graph: Graph
        The graph in which we run our simulation.

    start_node: int
        Index of the starting vertex, from which we are looking for the shortest path.

    end_node: int
        Index of the end vertex, towards which we are looking for the shortest path.

    current_path: list
        The list of virtual links to be used in the current path.
    """
    for index in range(0, main_graph.physical_distance(start_node, end_node) + 1):

        node_to_update = (start_node + index -1) % len(main_graph.vertices) + 1
        main_graph.vertices[node_to_update].local_knowledge.remove_from_local_knowledge(current_path)


def update_local_knowledge(main_graph, current_path: list, propagation_radius: int = 0):
    """
    Update the local knowledge of nodes within a specified propagation radius.

    Parameters
    ----------
    main_graph: Graph
        The graph in which we run our simulation.

    current_path: list
        The list of virtual links to be used in the current path.

    propagation_radius: int
        Index of the end vertex, towards which we are looking for the shortest path.
    """
    start_node = current_path[0]
    end_node = current_path[-1]

    # Determining which way will the local knowledge be propagated
    # Swap the start and end nodes, if we propagate along the shorter path
    if (end_node - start_node) % len(main_graph.vertices) > (start_node - end_node) % len(main_graph.vertices):
        start_node, end_node = end_node, start_node

    # Update the nodes along the physical graph about the virtual links used in the current path
    update_along_physical_graph(main_graph, start_node, end_node, current_path)

    # Further propagation based on the radius
    node_before = (start_node - propagation_radius - 1) % 32 + 1
    update_along_physical_graph(main_graph, node_before, start_node - 1, current_path)

    node_after = (end_node + propagation_radius - 1) % 32 + 1
    update_along_physical_graph(main_graph, end_node, node_after, current_path)


def local_knowledge_algorithm(graph_edges: list, source_destination_pairs: list, propagation_radius: int = 0,
                              exponential_scale: bool = True):
    """
    Runs the local knowledge algorithm specified by a propagation radius.
    The local knowledge of nodes about the virtual links used within a path is updated within
    a specified propagation radius.

    Parameters
    ----------
    graph_edges: list
        The graph edges to be added as local knowledge to the vertices of the graph.

    source_destination_pairs: list
        Specifies the source-destination pairs for which the algorithm will be computed.

    propagation_radius: int
        Index of the end vertex, towards which we are looking for the shortest path.

    exponential_scale: bool
        Specifies whether long link creation scales exponentially or polynomially with time.
    """

    # Generate the specific graph object
    main_graph = create_graph_with_local_knowledge(graph_edges)

    result_for_source_destination = []
    for sd_pair in source_destination_pairs:

        temp_result: tuple = ()

        simulation_settings = routing_simulation.Settings()

        # Initialize path
        # Determine shortest path based on local knowledge
        current_path = shortest_path.dijkstra(main_graph.vertices[sd_pair[0]].local_knowledge, sd_pair[0], sd_pair[1])
        current_distance = len(current_path)-1

        temp_result += (distribute_entanglement(main_graph, current_path, exponential_scale),)

        # Update local knowledge of the nodes that are along the current path
        update_local_knowledge(main_graph, current_path, propagation_radius)

        temp_result += (main_graph.get_sum_of_link_capacities(),)
        temp_result += (main_graph.get_available_link_count(),)
        temp_result += (current_distance,)
        result_for_source_destination.append(temp_result)
    return helper.map_tuple_gen(np.mean, zip(*result_for_source_destination))


def initial_knowledge_algorithm(main_graph, source_destination_pairs: list,
                                link_prediction: bool = False, exponential_scale: bool = True) -> tuple:
    """
    Runs the initial knowledge algorithm.
    The local knowledge of nodes about the virtual links used within a path is not updated.

    Parameters
    ----------
    main_graph: Graph
        The graph in which we run our simulation.

    source_destination_pairs: list
        Specifies the source-destination pairs for which the algorithm will be computed.

    link_prediction: bool
        Value determining whether or not link prediction is used.

    exponential_scale: bool
        Specifies whether long link creation scales exponentially or polynomially with time.
    """

    # Initialize paths in advance, then processing them one by one
    # The change in network is not considered in this approach (path is NOT UPDATED)
    path_store = initialize_paths(main_graph, source_destination_pairs, link_prediction=link_prediction)

    # Storing the distances of the paths
    distances = []
    for x in path_store:
        distances.append(len(x)-1)

    # Serving the demands in the quantum network
    # Calculating the entanglement delay times
    results = serve_demands(main_graph, path_store, exponential_scale)
    results += (distances,)

    return results


def initial_knowledge_step(main_graph, current_step: int, time_window_size: int,
                           number_of_source_destination_pairs: int, final_results: tuple,
                           link_prediction: bool = False) -> None:
    """
    Runs one step of the initial knowledge algorithm.
    The local knowledge of nodes about the virtual links used within a path is not updated.

    Parameters
    ----------
    main_graph: Graph
        The graph in which we run our simulation.

    current_step: int
        Specifies the step at which the initial knowledge algorithm is at.

    time_window_size: int
        Specifies the size of the time window used.

    number_of_source_destination_pairs: int
        Specifies the number of demands that need to be generated.

    final_results: tuple
        Tuple containing the final results for the simulation.

    link_prediction: bool
        Value determining whether or not link prediction is used.

    """
    step_in_time_window = current_step % time_window_size
    end_of_this_time_window = step_in_time_window == 0

    if end_of_this_time_window or current_step == number_of_source_destination_pairs:
        number_of_demands = time_window_size if end_of_this_time_window else step_in_time_window
        time_window_results = initial_knowledge_algorithm(main_graph, number_of_demands,
                                                          link_prediction=link_prediction)
        for x in range(len(time_window_results)):
            [final_results[x].append(element) for element in time_window_results[x]]

        # Update weights in the graph which might have been consumed
        if link_prediction:
            main_graph.update_stored_weights(current_step)

    return None


def initial_knowledge_init(graph_edges: list, source_destination_pairs: list, time_window_size: int = 1,
                           link_prediction: bool = False, exponential_scale: bool = True):
    """
    Initializes the initial knowledge algorithm.
    Create paths for the specified number of source and destination pairs, then send the packets along a specific path
    and store the waiting time and the distance

    Parameters
    ----------
    graph_edges: list
        The graph in which we run our simulation.

    source_destination_pairs: list
        Specifies the number of demands that need to be generated.

    time_window_size: int
        Specifies the size of the time window used.

    link_prediction: bool
        Value determining whether or not link prediction is used.

    exponential_scale: bool
        Specifies whether long link creation scales exponentially or polynomially with time.
    """

    number_of_measures = 4
    final_results = tuple([] for x in range(number_of_measures))
    main_graph = graph.Graph(graph_edges, link_prediction=link_prediction)
    number_of_source_destination_pairs = len(source_destination_pairs)

    if link_prediction:
        k = 1
        while k < number_of_source_destination_pairs + 1:
            initial_knowledge_step(main_graph, k, time_window_size, number_of_source_destination_pairs,
                                   final_results, link_prediction)
            k += 1
    else:
        final_results = initial_knowledge_algorithm(main_graph, source_destination_pairs,
                                                    link_prediction=link_prediction,
                                                    exponential_scale=exponential_scale)

    return helper.map_tuple_gen(np.mean, final_results)


def global_knowledge_algorithm(main_graph, source_destination_pairs: list,
                               exponential_scale: bool = True) -> list:
    """
    Applies the global knowledge approach for a certain graph by generating a specific number of demands.

    Parameters
    ----------
    main_graph : Graph
        The graph in which we serve the demands according to the global knowledge approach.

    source_destination_pairs: list
        Specifies the number of demands that need to be generated.

    exponential_scale: bool
        Specifies whether long link creation scales exponentially or polynomially with time.

    Notes
    ----------
    Add the data (measures) in the following order:
    (1) The waiting time
    (2) Number of available virtual links
    (3) Number of available edges
    (4) Distance of the path

    """
    result_for_source_destination = []
    for sd_pair in source_destination_pairs:
        temp_result = ()

        source, dest = sd_pair[0], sd_pair[1]

        # Initialize path
        # The change in network is considered in this approach (path is UPDATED)
        current_path = shortest_path.dijkstra(main_graph, source, dest)

        temp_result += (distribute_entanglement(main_graph, current_path, exponential_scale),)
        temp_result += (main_graph.get_sum_of_link_capacities(),)
        temp_result += (main_graph.get_available_link_count(),)
        temp_result += (len(current_path)-1,)
        result_for_source_destination.append(temp_result)
    return result_for_source_destination


def global_knowledge_init(graph_edges: list, source_destination_pairs: list,
                          exponential_scale: bool = True) -> tuple:
    """
    Initiates the global knowledge approach in graph.

    Parameters
    ----------
    graph_edges : list of tuple
        List of edges that specifies the edges of the graph to be created.

    source_destination_pairs: list
        Specifies the source-destination pairs.

    exponential_scale: bool
        Specifies whether long link creation scales exponentially or polynomially with time.

    """
    main_graph = graph.Graph(graph_edges)

    result_for_source_destination = global_knowledge_algorithm(main_graph, source_destination_pairs,
                                                               exponential_scale)
    return helper.map_tuple_gen(np.mean, zip(*result_for_source_destination))