import random
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../lib")

import routing_simulation
import helper
import graph

from collections import deque
import heapq


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


# Calculating tentative cost
# Using hardcoded value of 1 as a cost (unique costs need to be stored for each edge)

# Using different cost function if the capacity is 0 (cost is defined as 1000)

def weight(main_graph, start: int, end: int):
    if main_graph.get_edge_capacity(start, end) == 0:
        return routing_simulation.Settings().long_link_cost * main_graph.dist(start_node = start, end_node = end)
    else:
        return routing_simulation.Settings().original_cost


# The Dijkstra algorithm with a support for rebuilding the best next hop
def dijkstra(graph, start: int, finish: int) -> list:
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

            tentative_cost = distance[current] + weight(graph, current, child)

            if child not in distance.keys() or distance[child] > tentative_cost:
                distance[child] = tentative_cost
                parents[child] = current
                if parents[child] is None:
                    a = 2
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


def entanglement_swap(graph, start_node, end_node):
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


def distribute_entanglement(graph, current_path_list: list):
    # Initializing entanglement delay time
    edt = 0
    no_link_dist = 0
    remainder_of_path = deque(current_path_list)
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
                if routing_simulation.Settings().time_threshold <\
                        ((1 / routing_simulation.Settings().rebuild_probability) ** no_link_dist):
                    edt = (1 / routing_simulation.Settings().rebuild_probability) ** graph.dist(initial_node, end_node)
                else:
                    edt += ((1 / routing_simulation.Settings().rebuild_probability) ** no_link_dist)

            # Rebuild the missing virtual links based on the elapsed time
            # graph.update_edge_rebuild_times(edt)
            return edt

        # Put the end node back into the deque
        remainder_of_path.appendleft(end_node)


# Distributing entanglement based on the generated source destination pairs
# Processes these pairs by calling the distribute_entanglement method on the next path
# Distributes entanglement for each of the paths stored in the deque and pushes the result edt into a store
def get_simulation_data_for_paths(graph, paths: deque):
    edt_store = []
    virtual_links_store = []
    edge_store = []

    while True:

        current_path = paths.popleft()
        edt_store.append(distribute_entanglement(graph, current_path))
        virtual_links_store.append(graph.available_virtual_link_count())
        edge_store.append(graph.available_edge_count())

        if len(paths) == 0:
            return edt_store, virtual_links_store, edge_store


# Generates random source-destination pairs
# The number of source-destination pairs is given by the argument
def gen_rand_pairs(number_of_pairs: int):
    result = []
    for x in range(number_of_pairs):
        source = random.randint(1, routing_simulation.Settings().number_of_nodes)
        dest = random.randint(1, routing_simulation.Settings().number_of_nodes)
        while source == dest:
            dest = random.randint(1, routing_simulation.Settings().number_of_nodes)
        result += [[source, dest]]
    return result


# 1. Generates source-destination pairs
# 2. Finds the nodes in between the SD pairs by calling on the shortest path method
def initialize_paths(graph, number_of_source_destination_pairs: int):
    # Generate random pairs of nodes between which we are seeking a path
    randPairs = gen_rand_pairs(number_of_source_destination_pairs)

    # Assemble paths into one deque
    paths = deque()
    for pair in randPairs:
        path = dijkstra(graph, pair[0], pair[1])
        paths.appendleft(path)
    paths = paths
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


# ------------------------------------
# Implementing the local algorithms
# Nodes have knowledge about the graph based on the path discovery that they have participated in
def local_knowledge_algorithm(graph_edges: list, number_of_source_destination_pairs: int, knowledge_radius: int = 0):

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

        temp_result += (distribute_entanglement(main_graph, current_path),)

        # Update local knowledge of the nodes that are along the current path
        update_local_knowledge(main_graph, current_path, knowledge_radius)

        temp_result += (main_graph.available_virtual_link_count(),)
        temp_result += (main_graph.available_edge_count(),)
        temp_result += (current_distance,)
        result_for_source_destination.append(temp_result)
    return helper.map_tuple_gen(helper.mean, zip(*result_for_source_destination))


# Create paths for the specified number of source and destination pairs, then send the packets along a specific path
# and store the waiting time and the distance
# graph: the graph in which we send the packets
# number_of_source_destination_pairs: number of source and destination pairs for which we are creating a path
def initial_knowledge_init(graph_edges: list, number_of_source_destination_pairs: int):

    # Generate the specific graph object
    main_graph = graph.Graph(graph_edges)

    # Initialize paths in advance, then processing them one by one
    # The change in network is not considered in this approach (path is NOT UPDATED)
    path_store = initialize_paths(main_graph, number_of_source_destination_pairs)

    # Storing the distances of the paths
    distance_store = []
    for x in path_store:
        distance_store.append(len(x)-1)

    # Calculating the entanglement delay times
    results: tuple = get_simulation_data_for_paths(main_graph, path_store)
    results += (distance_store,)
    return helper.map_tuple_gen(helper.mean, results)


def global_algo(graph_edges: list, number_of_source_destination_pairs: int):
    # Generate the specific graph object
    main_graph = graph.Graph(graph_edges)

    result_for_source_destination = []
    for x in range(1, number_of_source_destination_pairs + 1):
        temp_result = ()

        simulation_settings = routing_simulation.Settings()

        source = random.randint(1, simulation_settings.number_of_nodes)
        dest = random.randint(1, simulation_settings.number_of_nodes)

        while source == dest:
            dest = random.randint(1, simulation_settings.number_of_nodes)

        # Initialize path
        # The change in network is considered in this approach (path is UPDATED)
        current_path = deque(dijkstra(main_graph, source, dest))

        # Add the data (measures) in the following order:
        # (1) The waiting time
        # (2) Number of available virtual links
        # (3) Number of available edges
        # (4) Distance of the path
        temp_result += (distribute_entanglement(main_graph, current_path),)
        temp_result += (main_graph.available_virtual_link_count(),)
        temp_result += (main_graph.available_edge_count(),)
        temp_result += (len(current_path)-1,)
        result_for_source_destination.append(temp_result)
    return helper.map_tuple_gen(helper.mean, zip(*result_for_source_destination))
