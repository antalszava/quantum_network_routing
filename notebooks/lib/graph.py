import math
from collections import deque

class Vertex:
    def __init__(self, vertex):
        self.name = vertex
        self.neighbours = {}
        self.local_knowledge = None

    # Storing the indices of neighbouring vertices
    def add_neighbour(self, neighbour, capacity=1):
        if neighbour not in self.neighbours:
            self.neighbours[neighbour] = capacity
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

    #@property
    def dist(self, start_node, end_node):
        #if start_node in self.Vertices.keys() and end_node in self.Vertices.keys():
        return min((start_node - end_node) % len(self.vertices), (end_node - start_node) % len(self.vertices))

    def add_vertex(self, vertex):
        if isinstance(vertex, Vertex):
            self.Vertices[vertex.name] = vertex.neighbours

    def add_vertices(self, vertices):
        for vertex in vertices:
            self.add_vertex(vertex)

    def add_edge(self, vertex_from, vertex_to):
        if isinstance(vertex_from, Vertex) and isinstance(vertex_to, Vertex):
            vertex_from.add_neighbour(vertex_to)
            if isinstance(vertex_from, Vertex) and isinstance(vertex_to, Vertex):
                self.Vertices[vertex_from.name] = vertex_from.neighbours
                self.Vertices[vertex_to.name] = vertex_to.neighbours

    def add_edges(self, edges):
        for edge in edges:
            self.add_edge(edge[0], edge[1])

    def add_capacity(self, start_node: int, end_node: int, capacity: int):
        try:
            self.vertices[start_node]
        except KeyError:
            log.debug("No such start node found among the vertices.")
        try:
            self.vertices[start_node].neighbours[end_node]
        except KeyError:
            log.debug("No such end node found among the vertices.")
        self.vertices[start_node].neighbours[end_node] += 1

    def get_edge_capacity(self, start_node, end_node):

        try:
            self.vertices[start_node]
        except KeyError:
            log.debug("No such start node found among the vertices.")
        try:
            self.vertices[start_node].neighbours[end_node]
        except KeyError:
            log.debug("No such end node found among the vertices.")
        return self.vertices[start_node].neighbours[end_node]

    def remove_virtual_link(self, start_node, end_node):

        if self.get_edge_capacity(start_node, end_node) != 0:
            self.vertices[start_node].neighbours[end_node] -= 1
        if self.get_edge_capacity(end_node, start_node) != 0:
            self.vertices[end_node].neighbours[start_node] -= 1

    def available_virtual_link_count(self):
        available_virtual_links_count = 0
        for start in self.vertices.keys():
            available_virtual_links_count += sum([capacity for capacity
                                                  in self.vertices[start].neighbours.values() if capacity != 0])
        if self.bidirectional:
            return available_virtual_links_count / 2
        else:
            return available_virtual_links_count

    def available_edge_count(self):
        available_edges_count = 0
        for start in self.vertices.keys():
            available_edges_count += sum([1 for capacity
                                          in self.vertices[start].neighbours.values() if capacity != 0])
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