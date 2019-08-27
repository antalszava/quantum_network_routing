import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../lib")

import heapq
import routing_simulation


def traceback_path(target, parents) -> list:
    path: list = []
    while target:
        path.append(target)
        target = parents[target]
    path = path
    return path


class HeapEntry:
    """
    Class for realising the heap entry class used in the Dijkstra algorithm

    Attributes
    ----------
    node: int
        The node from which the heap is built up.
    distance: float
        The distance specified for the initial node

    Notes
    -----
        The weight depends on the current capacity of the edge. We have to rebuild, if there are no more links available
        along the edge.
    """
    def __init__(self, node: int, distance: float):
        """
        The relation operation for heap objects based on the distance.

        Parameters
        ----------
        node: int
            The node from which the heap is built up.
        distance: float
            The distance specified for the initial node
        """
        self.node = node
        self.distance = distance

    def __lt__(self, other):
        """
        The relation operation for heap objects based on the distance.
        """
        return self.distance < other.distance


def weight(main_graph, start_node: int, end_node: int) -> int:
    """
    Calculating the tentative weight to a certain edge in the graph according to the link prediction rule
    while solving the shortest path problem with Dijkstra's algorithm.

    Parameters
    ----------
    main_graph: Graph
        The graph in which we want to assign weight in.
    start_node: int
        Index of the starting vertex of the edge.
    end_node: int
        Index of the end vertex of the edge.

    Notes
    -----
        The weight depends on the current capacity of the edge. We have to rebuild, if there are no more links available
        along the edge.
    """

    if main_graph.get_edge_capacity(start_node, end_node) == 0:
        return routing_simulation.Settings().long_link_cost *\
               main_graph.physical_distance(start_node=start_node, end_node=end_node)
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
    """
    if start == shortest_path_source or end == shortest_path_source:
        return weight(main_graph, start, end)
    else:
    """
    return main_graph.get_stored_weight_of_edge(start, end)


def dijkstra(graph, start_node: int, end_node: int, link_prediction: bool = False) -> list:
    """
    Finding the shortest path between a starting vertex and an end vertex.
    The Dijkstra algorithm with a heap construct is used, once the chain of vertices are determined for the shortest
    paths, a traceback is performed to produce the series of vertices comprising the shortest path to the
    end node specified.

    Parameters
    ----------
    graph: Graph
        The graph in which we want to assign weight in.
    start_node: int
        Index of the starting vertex, from which we are looking for the shortest path.
    end_node: int
        Index of the end vertex, towards which we are looking for the shortest path.
    link_prediction: bool
        A value determining whether or not we are applying link prediction

    Returns
    -----
        List of the vertices along the shortest path between the start and the end vertex.

    Notes
    -----
        If the source vertex has knowledge about the current edge, then we assign the real weight.

    """
    # The Dijkstra algorithm with a support for rebuilding the best next hop
    open_nodes = [HeapEntry(start_node, 0.0)]
    closed_nodes = set()
    parents = {start_node: None}
    distance = {start_node: 0.0}

    # Take all the nodes that are still to be processed
    while open_nodes:
        current = heapq.heappop(open_nodes).node

        # If the current node is the end node, then we are finished
        if current is end_node:
            return traceback_path(end_node, parents)

        # We continue, if we have already processed the current node
        if current in closed_nodes:
            continue

        closed_nodes.add(current)

        # For every child of the current node (taking the neighbours)
        # We continue, if we have already processed them
        # Else we process it based on the options specified for the algorithm
        for child in graph.vertices[current].neighbours.keys():
            if child in closed_nodes:
                continue

            current_weight = weight(graph, current, child)\
                if not link_prediction else link_prediction_weight(graph, current, child, start_node)

            tentative_cost = distance[current] + current_weight

            if child not in distance.keys() or distance[child] > tentative_cost:
                distance[child] = tentative_cost
                parents[child] = current
                heap_entry = HeapEntry(child, tentative_cost)
                heapq.heappush(open_nodes, heap_entry)

    return traceback_path(end_node, parents)