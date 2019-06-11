import random
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../lib")

import routing_simulation
import helper
import graph

from collections import deque
import heapq
import math

def traceback_path(target, parents) -> list:
    path: list = []
    while target:
        path.append(target)
        target = parents[target]
    path = path
    return path


class HeapEntry:
    def __init__(self, node, distance):
        self.node = node
        self.distance = distance

    def __lt__(self, other):
        return self.distance < other.distance


def weight(main_graph, start: int, end: int) -> int:
    """
    Calculating the tentative weight to a certain edge in the graph according to the link prediction rule
    while solving the shortest path problem with Dijkstra's algorithm.

    Parameters
    ----------
    main_graph: Graph
        The graph in which we want to assign weight in.
    start: int
        Index of the starting vertex of the edge.
    end: int
        Index of the end vertex of the edge.

    Notes
    -----
        The weight depends on the current capacity of the edge. We have to rebuild, if there is no more links available
        along the edge.
    """

    if main_graph.get_edge_capacity(start, end) == 0:
        return routing_simulation.Settings().long_link_cost * main_graph.dist(start_node=start, end_node=end)
    else:
        return routing_simulation.Settings().original_cost


def link_prediction_weight(main_graph, start: int, end: int, shortest_path_source: int) -> int:
    """
    Calculating the tentative weight to a certain edge in the graph according to the link prediction rule
    while solving the shortest path problem with Dijkstra's algorithm.

    Parameters
    ----------
    main_graph: Graph
        The graph in which we want to assign weight in.
    start: int
        Index of the starting vertex of the edge.
    end: int
        Index of the end vertex of the edge.
    shortest_path_source: int
        Index of the source vertex to which we want to solve the shortest path problem.

    Notes
    -----
        If the source vertex has knowledge about the current edge, then we assign the real weight.

    """
    if start == shortest_path_source or end == shortest_path_source:
        return weight(main_graph, start, end)
    else:
        return main_graph.get_stored_weight_of_edge(start, end)


# The Dijkstra algorithm with a support for rebuilding the best next hop
def dijkstra(graph, start: int, finish: int, link_prediction: bool = False) -> list:
    open_nodes = [HeapEntry(start, 0.0)]
    closed_nodes = set()
    parents = {start: None}
    distance = {start: 0.0}

    while open_nodes:
        current = heapq.heappop(open_nodes).node

        if current is finish:
            return traceback_path(finish, parents)

        if current in closed_nodes:
            continue

        closed_nodes.add(current)

        # For every child of the current node (taking the neighbours)
        for child in graph.vertices[current].neighbours.keys():
            if child in closed_nodes:
                continue

            current_weight = weight(graph, current, child)\
                if not link_prediction else link_prediction_weight(graph, current, child, start)

            tentative_cost = distance[current] + current_weight

            if child not in distance.keys() or distance[child] > tentative_cost:
                distance[child] = tentative_cost
                parents[child] = current
                heap_entry = HeapEntry(child, tentative_cost)
                heapq.heappush(open_nodes, heap_entry)

    return traceback_path(finish, parents)


'''
Processes a source-destination pair of distance one of the current path
by checking the capacity of the link between the start and the end node

If the capacity is 0, then a probabilistic rebuild approach is used

Returns the elapsed time that was needed to process the particular source-destination pair

If the capacity of the link is 0, then
do probabilistic rebuilding

Otherwise: 
Consumes a link from the remaining ones
Alternatively: add the threshold waiting time for rebuilding
'''


def entanglement_swap(graph, start_node:int , end_node:int ) -> tuple:
    local_edt = 0
    local_no_link_dist = 0

    if graph.get_edge_capacity(start_node, end_node) == 0:

        local_no_link_dist += graph.dist(start_node, end_node)
    else:

        # Remove the link between startNode and endNode
        graph.remove_virtual_link(start_node, end_node)

        # Incrementing the entanglement delay time
        local_edt += 1

    return local_edt, local_no_link_dist


'''
# Works through a source-destination pair by traversing through the nodes in between and adding the elapsed time
#
# Calls on the entanglement_swap method as many times as big the distance between the source and the destination is
'''


def distribute_entanglement(graph, current_path: list, exponential_scale: bool = True):
    # Initializing entanglement delay time
    edt = 0
    no_link_dist = 0
    remainder_of_path = deque(current_path)
    initial_node = remainder_of_path.popleft()
    get_initial = True

    # Take the leftmost two nodes out of the deque and get the edt until we are finished

    while True:
        if get_initial:
            start_node = initial_node
            get_initial = False
        else:
            start_node = remainder_of_path.popleft()
        end_node = remainder_of_path.popleft()

        # Calculate the edt for the current step
        local_temp1, local_temp2 = entanglement_swap(graph, start_node, end_node)
        edt += local_temp1
        no_link_dist += local_temp2

        # Check if we have processed the path
        if len(remainder_of_path) == 0:

            # Check if there are links which were not available through the path
            if no_link_dist > 0:

                # If we cannot create the missing entangled links in the specific threshold time
                # Then simply generate entangled links along the physical graph
                local_settings = routing_simulation.Settings()
                successful_rebuild_time = 1 / local_settings.rebuild_probability

                time_to_rebuild_path = successful_rebuild_time ** no_link_dist\
                    if exponential_scale else no_link_dist ** 2

                if local_settings.time_threshold < time_to_rebuild_path:
                    if exponential_scale:
                        edt = successful_rebuild_time ** graph.dist(initial_node, end_node)
                    else:
                        edt = graph.dist(initial_node, end_node) ** 2
                else:
                    edt += time_to_rebuild_path

            # Rebuild the missing virtual links based on the elapsed time
            # graph.update_edge_rebuild_times(edt)
            return edt

        # Put the end node back into the deque
        remainder_of_path.appendleft(end_node)


# Distributing entanglement based on the generated source destination pairs
# Processes these pairs by calling the distribute_entanglement method on the next path
# Distributes entanglement for each of the paths stored in the deque and pushes the result edt into a store
def serve_demands(graph, paths: deque, exponential_scale: bool = True) -> tuple:
    edt_store = []
    virtual_links_store = []
    edge_store = []

    while True:

        current_path = paths.popleft()
        edt_store.append(distribute_entanglement(graph, current_path, exponential_scale))
        virtual_links_store.append(graph.available_virtual_link_count())
        edge_store.append(graph.available_edge_count())

        if len(paths) == 0:
            return edt_store, virtual_links_store, edge_store


def generate_demand(number_of_nodes: int) -> tuple:
    """
    Generates a random source and destination pair in

    Parameters
    ----------
    number_of_nodes : int
        Integer specifying the number of nodes in the graph.
    """
    source = random.randint(1, number_of_nodes)
    dest = random.randint(1, number_of_nodes)
    while source == dest:
        dest = random.randint(1, number_of_nodes)
    return source, dest


def gen_rand_pairs(number_of_pairs: int) -> list:
    """
    Generates a certain number of random source-destination pairs.

    Parameters
    ----------
    number_of_pairs : int
        Integer specifying the number of source-destination pairs to be generated.
    """
    result = []
    number_of_nodes = routing_simulation.Settings().number_of_nodes

    for x in range(number_of_pairs):
        result += [generate_demand(number_of_nodes)]
    return result


# 1. Generates source-destination pairs
# 2. Finds the nodes in between the SD pairs by calling on the shortest path method
def initialize_paths(graph, number_of_source_destination_pairs: int, link_prediction: bool = False) -> deque:
    # Generate random pairs of nodes between which we are seeking a path
    random_pairs = gen_rand_pairs(number_of_source_destination_pairs)

    # Assemble paths into one deque
    paths = deque()
    for pair in random_pairs:
        path = dijkstra(graph, pair[0], pair[1], link_prediction=link_prediction)
        paths.appendleft(path)
    return paths


def create_graph_with_local_knowledge(graph_edges: list):
    temp_graph = graph.Graph(graph_edges)
    temp_graph.add_local_knowledge(graph_edges)
    return temp_graph


def update_along_physical_graph(main_graph, start_node: int, end_node: int, current_path: list):
    for index in range(0, main_graph.dist(start_node, end_node) + 1):

        node_to_update = (start_node + index -1) % len(main_graph.vertices) + 1
        main_graph.vertices[node_to_update].local_knowledge.remove_from_local_knowledge(current_path)


# Updates the local knowledge of the nodes along the physical graph
def update_local_knowledge(main_graph, current_path: list, knowledge_radius: int = 0):
    start_node = current_path[0]
    end_node = current_path[-1]

    # Determining which way will the local knowledge be propagated
    # Swap the start and end nodes, if we propagate along the shorter path
    if (end_node - start_node) % len(main_graph.vertices) > (start_node - end_node) % len(main_graph.vertices):
        start_node, end_node = end_node, start_node

    # Update the nodes along the physical graph about the virtual links used in the current path
    update_along_physical_graph(main_graph, start_node, end_node, current_path)

    # Further propagation based on the radius
    node_before = (start_node - knowledge_radius - 1) % 32 + 1
    update_along_physical_graph(main_graph, node_before, start_node - 1, current_path)

    node_after = (end_node + knowledge_radius - 1) % 32 + 1
    update_along_physical_graph(main_graph, end_node, node_after, current_path)


# Nodes have knowledge about the graph based on the path discovery that they have participated in
def local_knowledge_algorithm(graph_edges: list, number_of_source_destination_pairs: int, knowledge_radius: int = 0,
                              exponential_scale: bool = True):

    # Generate the specific graph object
    main_graph = create_graph_with_local_knowledge(graph_edges)

    result_for_source_destination = []
    for x in range(1, number_of_source_destination_pairs + 1):

        temp_result: tuple = ()

        source = random.randint(1, routing_simulation.Settings().number_of_nodes)
        dest = random.randint(1, routing_simulation.Settings().number_of_nodes)

        while source == dest:
            dest = random.randint(1, routing_simulation.Settings().number_of_nodes)

        # Initialize path
        # Determine shortest path based on local knowledge
        current_path = dijkstra(main_graph.vertices[source].local_knowledge, source, dest)
        current_distance = len(current_path)-1

        temp_result += (distribute_entanglement(main_graph, current_path, exponential_scale),)

        # Update local knowledge of the nodes that are along the current path
        update_local_knowledge(main_graph, current_path, knowledge_radius)

        temp_result += (main_graph.available_virtual_link_count(),)
        temp_result += (main_graph.available_edge_count(),)
        temp_result += (current_distance,)
        result_for_source_destination.append(temp_result)
    return helper.map_tuple_gen(helper.mean, zip(*result_for_source_destination))


def initial_knowledge_algorithm(main_graph, number_of_source_destination_pairs: int,
                                link_prediction: bool = False, exponential_scale: bool = True) -> tuple:

    # Initialize paths in advance, then processing them one by one
    # The change in network is not considered in this approach (path is NOT UPDATED)
    path_store = initialize_paths(main_graph, number_of_source_destination_pairs, link_prediction=link_prediction)

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


# Create paths for the specified number of source and destination pairs, then send the packets along a specific path
# and store the waiting time and the distance
# graph: the graph in which we send the packets
# number_of_source_destination_pairs: number of source and destination pairs for which we are creating a path
def initial_knowledge_init(graph_edges: list, number_of_source_destination_pairs: int, time_window_size: int = 5,
                           link_prediction: bool = False, exponential_scale: bool = True):

    number_of_measures = 4
    final_results = tuple([] for x in range(number_of_measures))
    main_graph = graph.Graph(graph_edges, link_prediction=link_prediction)

    if link_prediction:
        k = 1
        while k < number_of_source_destination_pairs + 1:
            initial_knowledge_step(main_graph, k, time_window_size, number_of_source_destination_pairs,
                                   final_results, link_prediction)
            k += 1
    else:
        final_results = initial_knowledge_algorithm(main_graph, number_of_source_destination_pairs,
                                                    link_prediction=link_prediction,
                                                    exponential_scale=exponential_scale)

    return helper.map_tuple_gen(helper.mean, final_results)


def global_knowledge_algorithm(main_graph, number_of_source_destination_pairs: int,
                               exponential_scale: bool = True) -> list:
    """
    Applies the global knowledge approach for a certain graph by generating a specific number of demands.

    Parameters
    ----------
    main_graph : list of tuple
        The graph in which we serve the demands according to the global knowledge approach.

    number_of_source_destination_pairs: bool
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
    number_of_nodes = routing_simulation.Settings().number_of_nodes
    for x in range(1, number_of_source_destination_pairs + 1):
        temp_result = ()

        source, dest = generate_demand(number_of_nodes)

        # Initialize path
        # The change in network is considered in this approach (path is UPDATED)
        current_path = dijkstra(main_graph, source, dest)

        temp_result += (distribute_entanglement(main_graph, current_path, exponential_scale),)
        temp_result += (main_graph.available_virtual_link_count(),)
        temp_result += (main_graph.available_edge_count(),)
        temp_result += (len(current_path)-1,)
        result_for_source_destination.append(temp_result)
    return result_for_source_destination


def global_knowledge_init(graph_edges: list, number_of_source_destination_pairs: int,
                          exponential_scale: bool = True) -> tuple:
    """
    Initiates the global knowledge approach in graph.

    Parameters
    ----------
    graph_edges : list of tuple
        Edgelist that specifies the edges of the graph to be created.

    number_of_source_destination_pairs: bool
        Specifies the number of demands that need to be generated.

    exponential_scale: bool
        Specifies whether long link creation scales exponentially or polynomially with time.

    """
    main_graph = graph.Graph(graph_edges)

    result_for_source_destination = global_knowledge_algorithm(main_graph, number_of_source_destination_pairs,
                                                               exponential_scale)
    return helper.map_tuple_gen(helper.mean, zip(*result_for_source_destination))
