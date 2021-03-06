import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../lib")
import routing_simulation
import shortest_path
import helper

from collections import deque


class Edge:
    def __init__(self, capacity: int = 1, weight: int = 1):
        """
        Class for an edge.

        Parameters
        ----------
        capacity: int
            Specifies the maximum and the current capacity of the edge

        weight: int
            Specifies the weight of the edge

        Attributes
        ----------
        capacity: int
            Current capacity of the edge

        capacity: int
            Maximum capacity of the edge

        weight: int
            Weight of the edge
        """
        self.capacity = capacity
        self.max_capacity = capacity
        self.weight = weight

    def decrement_capacity(self):
        """
        Decrement the capacity until it is 0.
        """
        if self.capacity > 0:
            self.capacity -= 1


class Vertex:
    def __init__(self, index):
        """
        Class for a vertex.

        Parameters
        ----------
        index: int
            Identifier of the vertex

        Attributes
        ----------
        neighbours: dict
            Dictionary for vertex objects of neighbours of self.

        local_knowledge: Graph
            Local knowledge about the graph in which self is a vertex.
        """
        self.index = index
        self.neighbours = {}
        self.local_knowledge = None

    def add_neighbour(self, neighbour, capacity: int = 1, weight: int = 1):
        """
        Adding a neighbour by creating an edge between self and the specified neighbour vertex.
        The capacity and the weight of the edge can be specified.

        Parameters
        ----------
        neighbour: Vertex
            Vertex that is added as a neighbour to self.

        capacity: int
            Capacity of the edge added between self and the neighbour.

        weight: int
            Weight of the edge added between self and the neighbour.

        """
        if neighbour not in self.neighbours:
            self.neighbours[neighbour] = Edge(capacity, weight)
        else:
            return False

    def add_local_knowledge(self, graph):
        """
        Adding local knowledge to the vertex about the state of the graph.

        Parameters
        ----------
        graph: Graph

        """
        self.local_knowledge = graph

    def __repr__(self):
        return str(self.neighbours)


class Graph:

    def __init__(self, edges: list = None, link_prediction=False):
        """
        Class for an undirected graph specified by a list of tuples.
        Made up of the instantiations of Vertex and Edge classes.

        Parameters
        ----------
        edges : list of tuples
            Edgelist that specifies the starting vertex, end vertex and the capacity of the link.

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
        """

        self.Vertices = {}
        self.edge_frequencies = {}
        self.link_consumption_time = {}

        # Initializing graph based on edges
        if edges is not None:

            # Adding undirected links
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
        """
        Property for the private attribute Vertices.
        """
        return self.Vertices

    def physical_distance(self, start_node: int, end_node: int):
        """
        Returns the distance between two vertices specified by their indices.
        We assume that there is a connected one-dimensional lattice as a physical graph underlying the virtual graph.

        Parameters
        ----------
        start_node: int
            Index of the starting vertex.
        end_node: int
            Index of the end vertex.

        """
        return min((start_node - end_node) % len(self.vertices), (end_node - start_node) % len(self.vertices))

    def add_capacity(self, start_node: int, end_node: int, capacity: int):
        """
        Returns the distance between two vertices specified by their indices.
        It is assumed that the graph is a one-dimensional lattice and is connected.

        Parameters
        ----------
        start_node: int
            Index of the starting vertex.
        end_node: int
            Index of the end vertex.
        capacity: int
            Value to increase the capacity with

        """
        if start_node not in self.vertices or end_node not in self.vertices:
            print("No such nodes found as starts of an edge in the graph.")
        elif end_node not in self.vertices[start_node].neighbours or \
                start_node not in self.vertices[end_node].neighbours:
            print("No such nodes found as ends of an edge in the graph.")
        else:
            self.vertices[start_node].neighbours[end_node].capacity += capacity

    def get_stored_weight_of_edge(self, start_node: int, end_node: int) -> int:
        """
        Return the weight of the link specified by a start and an end node.

        Parameters
        ----------
        start_node: int
            Index of the starting vertex.
        end_node: int
            Index of the end vertex.

        Returns
        -------
        weight: int
            Weight of the edge determined by the start and end nodes.
        """
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
        elif end_node not in self.vertices[start_node].neighbours or\
            start_node not in self.vertices[end_node].neighbours:
            print("No such nodes found as starts of an edge in the graph.")
        else:
            self.vertices[start_node].neighbours[end_node].weight = weight
            self.vertices[end_node].neighbours[start_node].weight = weight

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

            new_weight = successful_rebuild_time ** self.physical_distance(start_node, end_node)

            self.update_stored_weight_of_edge(start_node, end_node, new_weight)

    def get_edge_capacity(self, start_node: int, end_node: int):
        """
        Get capacity of an edge determined by start and end nodes.

        Parameters
        ----------
        start_node: int
            Index of the starting vertex of the edge.
        end_node: int
            Index of the end vertex of the edge.

        Returns
        -------
        capacity: int
            Capacity of the edge determined by the start and end nodes.
        """
        if start_node not in self.vertices or end_node not in self.vertices:
            print("No such nodes found as starts of an edge in the graph.")
        elif end_node not in self.vertices[start_node].neighbours or\
            start_node not in self.vertices[end_node].neighbours:
            print("No such nodes found as starts of an edge in the graph.")
        else:
            return self.vertices[start_node].neighbours[end_node].capacity

    def remove_virtual_link(self, start_node: int, end_node: int):
        """
        Removes a link from the virtual graph.

        Parameters
        ----------
        start_node: int
            Index of the starting vertex of the edge.
        end_node: int
            Index of the end vertex of the edge.
        """

        if self.get_edge_capacity(start_node, end_node) != 0:
            self.vertices[start_node].neighbours[end_node].decrement_capacity()

        if self.get_edge_capacity(end_node, start_node) != 0:
            self.vertices[end_node].neighbours[start_node].decrement_capacity()

    def get_sum_of_link_capacities(self):
        """
        Check the capacity of links and return the current number of available links in the graph.
        Multiple links between the same vertices are counted regarded separately.

        Returns
        -------
        capacity: int
            Capacity of the link determined by the start and end nodes.
        """
        link_capacity_sum = 0
        for start in self.vertices.keys():
            link_capacity_sum += sum([link.capacity
                                    for link in self.vertices[start].neighbours.values() if link.capacity != 0])
        return link_capacity_sum / 2

    def get_available_link_count(self):
        """
        Check the capacity of links and return the current number of available links in the graph.
        Multiple links between the same vertices are counted regarded separately.

        Returns
        -------
        capacity: int
            Capacity of the edge determined by the start and end nodes.
        """
        available_link_count = 0
        for start in self.vertices.keys():
            available_link_count += sum([1
                                         for edge in self.vertices[start].neighbours.values() if edge.capacity != 0])

        return available_link_count / 2

    def get_paths_for_all_pairs(self) -> list:
        """
        Get all the shortest paths for every possible source and destination pairs in a given graph.

        Returns:
            List of shortest paths for each viable pair.
        """
        return [shortest_path.dijkstra(self, x, y) for x in range(1, len(self.Vertices) + 1)
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

    def assign_edge_frequencies(self):
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
                return

    def initialize_link_consumption_times(self):
        """
        Using a link prediction approach, initialize a link consumption time for each of the edges.

        Serves as an expectatiion value of after what time the link will most probably not exist anymore.

        Returns:
            None
        """

        # Coefficient coming from all the possible edges in the graph
        link_consumption_coefficient = len(self.vertices)*(len(self.vertices)-1)/2

        for x in self.edge_frequencies:
            self.link_consumption_time[x] = link_consumption_coefficient/self.edge_frequencies[x]

    def add_local_knowledge(self, local_knowledge_graph_edges: list):
        """
        Adds local knowledge about the state of the graph to the vertices.

        Args:
            local_knowledge_graph_edges: list of the edges as local knowledge
        """
        for node in self.vertices:
            local_knowledge_graph = Graph(local_knowledge_graph_edges)
            self.Vertices[node].add_local_knowledge(local_knowledge_graph)

    def remove_from_local_knowledge(self, update_along_edges: list):
        """
        Removes edge from the local knowledge that have been consumed already.

        Args:
            update_along_edges: list of the edges that have been consumed and are to be removed
        """
        current_path = deque(update_along_edges)
        start_node = current_path.popleft()

        while len(current_path) > 0:
            end_node = current_path.popleft()
            self.remove_virtual_link(start_node, end_node)
            start_node = end_node
