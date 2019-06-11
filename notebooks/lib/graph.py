import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../lib")
import routing_simulation
import routing_algorithms
import helper

from collections import deque


class Edge:
    def __init__(self, capacity:int = 1, weight: int = 1):
        self.current_capacity = capacity
        self.max_capacity = capacity
        self.weight = weight
        self.rebuild_times = []

    def remove_edge(self):
        if self.current_capacity > 0:
            self.current_capacity -= 1

    def set_time_till_rebuild(self, time_till_rebuild):
        self.rebuild_times.append(time_till_rebuild)

    def update_time_till_rebuild(self, time_amount):
        self.rebuild_times = [(x - time_amount) for x in self.rebuild_times if (x - time_amount) > 0]
        new_current_capacity = self.max_capacity - len(self.rebuild_times)
        if new_current_capacity > self.current_capacity:
            self.current_capacity = new_current_capacity


class Vertex:
    def __init__(self, vertex):
        self.name = vertex
        self.neighbours = {}
        self.local_knowledge = None

    # Storing the indices of neighbouring vertices
    def add_neighbour(self, neighbour, capacity=1):
        if neighbour not in self.neighbours:
            self.neighbours[neighbour] = Edge(capacity, 1)
        else:
            return False

    def add_neighbours(self, neighbours):
        for neighbour in neighbours:
            if isinstance(neighbour, tuple):
                self.add_neighbour(neighbour[0], neighbour[1])
            else:
                self.add_neighbour(neighbour)

    def add_local_knowledge(self, graph):
        self.local_knowledge = graph

    def __repr__(self):
        return str(self.neighbours)


class Graph:

    # Check that the arguments are valid
    # edges: list of edges awaited in the format of list(tuple(start_node, end_node, capacity))
    def __init__(self, edges: list = None, bidirectional=True, link_prediction=False):
        """
        Generic class for a graph specified by an edgelist.

        Parameters
        ----------
        edges : list of tuple
            Edgelist that specifies the starting vertex, end vertex and the capacity of the link.

        bidirectional: bool
            Specifies whether or not we have bidirectional links in the graph.

        link_prediction: bool
            Specifies whether or not we are conducting link prediction on the graph.

        Attributes
        ----------
        Vertices: dict
            Dictionary for vertex objects of the graph.

        edge_frequencies: dict
            Dictionary for the frequency of edges in the shortest paths.

        link_consumption_time: dict
            Dictionary for the times of link consumption.

        bidirectional: bool
            Boolean specifying whether or not we have bidirectional links in the graph.

        Raises
        ------

        Notes
        -----


        Examples
        --------
        """

        self.Vertices = {}
        self.edge_frequencies = {}
        self.link_consumption_time = {}
        self.bidirectional = bidirectional

        # Initializing graph based on edges
        if edges is not None:
            # wrong_edges = [i for i in edges if len(i) not in [2, 4]]
            # if wrong_edges:
            #    raise ValueError('Wrong edges data: %s', wrong_edges)

            # Adding bidirectional links
            for (start_node, end_node, capacity) in edges:

                # Adding onward link
                if start_node not in self.Vertices.keys():
                    self.Vertices[start_node] = Vertex(start_node)

                self.Vertices[start_node].add_neighbour(end_node, capacity)

                # Adding backward link
                if end_node not in self.Vertices.keys():
                    self.Vertices[end_node] = Vertex(end_node)

                self.Vertices[end_node].add_neighbour(start_node, capacity)

            if link_prediction:
                self.assign_edge_frequencies()
                self.initialize_link_consumption_times()

    @property
    def vertices(self):
        return self.Vertices

    def vertex(self, vertex):
        try:
            self.vertices[vertex]
        except:
            log.debug("No such start node found among the vertices.")
        return self.vertices[vertex]

    # TO-DO:
    # Rewrite so that it works for lattices as well
    def dist(self, start_node, end_node):
        return min((start_node - end_node) % len(self.vertices), (end_node - start_node) % len(self.vertices))

    def add_capacity(self, start_node: int, end_node: int, capacity: int):
        try:
            self.vertices[start_node]
        except KeyError:
            log.debug("No such start node found among the vertices.")
        try:
            self.vertices[start_node].neighbours[end_node]
        except KeyError:
            log.debug("No such end node found among the vertices.")
        self.vertices[start_node].neighbours[end_node].current_capacity += capacity

    def get_stored_weight_of_edge(self, start_node: int, end_node: int) -> int:
        if start_node not in self.vertices or end_node not in self.vertices:
            print("No such nodes found as starts of an edge in the graph.")

        if end_node not in self.vertices[start_node].neighbours or\
            start_node not in self.vertices[end_node].neighbours:
            print("No such nodes found as ends of an edge in the graph.")
        return self.vertices[start_node].neighbours[end_node].weight

    def update_stored_weight_of_edge(self, start_node: int, end_node: int, weight: int) -> None:
        """
        Updates the stored weight of a certain edge by the value specified.

        Parameters
        ----------
        start_node: int
            Index of the starting vertex of the edge.
        end_node: int
            Index of the end vertex of the edge.
        weight: int
            The weight that is used during the update process.

        Notes
        -----
            If the source vertex has knowledge about the current edge, then we assign the real weight.

        """
        if start_node not in self.vertices or end_node not in self.vertices:
            print("No such nodes found as starts of an edge in the graph.")

        if end_node not in self.vertices[start_node].neighbours or\
            start_node not in self.vertices[end_node].neighbours:
            print("No such nodes found as starts of an edge in the graph.")
        self.vertices[start_node].neighbours[end_node].weight = weight
        self.vertices[end_node].neighbours[start_node].weight = weight
        return None

    def update_stored_weights(self, elapsed_time: int):
        """
        Determines those stored weights of edges in the graph whose link consumption time has been passed according to
        the elapsed time. Then updates these weights with a weight corresponding to the rebuild time of an unavailable
        link.

        Parameters
        ----------
        elapsed_time: int
            Elapsed time since the network has been serving demands.

        Notes
        -----
            This is the main part of functionality of the link prediction rule. After the elapsed time, some nodes just
            "act as if" they knew that links further away were missing.

        """
        edges_to_update = [x for x in self.edge_frequencies if self.link_consumption_time[x] < elapsed_time]
        for edge in edges_to_update:

            start_node = edge[0]
            end_node = edge[1]
            successful_rebuild_time = 1 / routing_simulation.Settings().rebuild_probability

            new_weight = successful_rebuild_time ** self.dist(start_node, end_node)

            self.update_stored_weight_of_edge(start_node, end_node, new_weight)

    def get_edge_capacity(self, start_node, end_node):

        try:
            self.vertices[start_node]
        except KeyError:
            log.debug("No such start node found among the vertices.")
        try:
            self.vertices[start_node].neighbours[end_node]
        except KeyError:
            log.debug("No such end node found among the vertices.")
        return self.vertices[start_node].neighbours[end_node].current_capacity

    def remove_virtual_link(self, start_node, end_node):

        if self.get_edge_capacity(start_node, end_node) != 0:
            self.vertices[start_node].neighbours[end_node].remove_edge()
            self.vertices[start_node].neighbours[end_node].set_time_till_rebuild(routing_simulation.Settings()
                                                                                 .time_threshold)
        if self.get_edge_capacity(end_node, start_node) != 0:
            self.vertices[end_node].neighbours[start_node].remove_edge()
            self.vertices[end_node].neighbours[start_node].set_time_till_rebuild(routing_simulation.Settings()
                                                                                 .time_threshold)

    def available_virtual_link_count(self):
        available_virtual_links_count = 0
        for start in self.vertices.keys():
            available_virtual_links_count += sum([edge.current_capacity for edge
                                                  in self.vertices[start].neighbours.values() if edge.current_capacity != 0])
        if self.bidirectional:
            return available_virtual_links_count / 2
        else:
            return available_virtual_links_count

    def get_paths_for_all_pairs(self) -> list:
        """
        Get all the shortest paths for every possible source and destination pairs in a given graph.

        Args:
            graph: (graph.Graph) Graph in which we determine the shortest paths

        Returns:
            List of shortest paths for each viable pair.
        """
        return [routing_algorithms.dijkstra(self, x, y) for x in range(1, len(self.Vertices) + 1)
                for y in range(1, len(self.Vertices) + 1) if x != y]

    def add_frequency_for_path(self, current_path: list) -> None:
        """
        Get the frequency of edges used in paths among all possible source-destination pairs

        Args:
            current_path: The second parameter.

        Returns:
            None as return value, as we are using the updated dictionary

        """
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

            if start_node > end_node:
                helper.add_tuple_to_dictionary(self.edge_frequencies, ((end_node, start_node), 1))
            else:
                helper.add_tuple_to_dictionary(self.edge_frequencies, ((start_node, end_node), 1))

            # Check if we have processed the path
            if len(remainder_of_path) == 0:
                return None

            # Put the end node back into the deque
            remainder_of_path.appendleft(end_node)

    def assign_edge_frequencies(self) -> None:
        """
        Get all the shortest paths for every possible source and destination pairs in a given graph.

        Args:
            graph: (graph.Graph) Graph in which we determine the shortest paths

        Returns:
            Dictionary containing frequencies of edges in graph, edges are specified by tuples
        """
        paths = deque(self.get_paths_for_all_pairs())
        while True:

            current_path = paths.popleft()
            self.add_frequency_for_path(current_path)

            if len(paths) == 0:
                return None

    def initialize_link_consumption_times(self) -> None:
        """
        Using a link prediction approach, initialize a link consumption time for each of the edges.

        Serves as an expectatiion value of after what time the link will most probably not exist anymore.

        Args:
            graph: (graph.Graph) Graph in which we determine the shortest paths

        Returns:
            None
        """

        # Coefficient derived from the number of nodes in the graph (all possible edges in the graph)
        link_consumption_coefficient = len(self.vertices)*len(self.vertices)/2

        for x in self.edge_frequencies:
            self.link_consumption_time[x] = link_consumption_coefficient/self.edge_frequencies[x]

        return None

    def available_edge_count(self):
        available_edges_count = 0
        for start in self.vertices.keys():
            available_edges_count += sum([1 for edge
                                          in self.vertices[start].neighbours.values() if edge.current_capacity != 0])
        if self.bidirectional:
            return available_edges_count / 2
        else:
            return available_edges_count
        
    def add_local_knowledge(self, local_knowledge_graph_edges: list):
        for node in self.vertices:
            local_knowledge_graph = Graph(local_knowledge_graph_edges)
            self.Vertices[node].add_local_knowledge(local_knowledge_graph)

    def remove_from_local_knowledge(self, update_along_edges: list):
        current_path = deque(update_along_edges)
        start_node = current_path.popleft()

        while len(current_path) > 0:
            end_node = current_path.popleft()
            self.remove_virtual_link(start_node, end_node)
            start_node = end_node

    def update_edge_rebuild_times(self, update_time):
        for start_node in self.Vertices.keys():
            for end_node in self.Vertices[start_node].neighbours.keys():
                self.Vertices[start_node].neighbours[end_node].update_time_till_rebuild(update_time)
