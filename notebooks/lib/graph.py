import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../lib")
import routing_simulation

from collections import deque


class Edge:
    def __init__(self, capacity=1):
        self.current_capacity = capacity
        self.max_capacity = capacity
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
            self.neighbours[neighbour] = Edge(capacity)
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
    def __init__(self, edges: list = None, bidirectional=True):
        self.Vertices = {}
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
