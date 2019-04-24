import numpy as np
import random
import math


def max_number_of_virtual_links(number_of_nodes: int = 32, max_distance_threshold = 4):
    max_long_link_type = int(math.log(max_distance_threshold, 2))
    return sum([int(number_of_nodes/2 ** x) for x in range(0, max_long_link_type + 1)])


class PhysicalGraph:
    def __init__(self, number_of_nodes: int = 32, original_capacity: int = 1):

        self.number_of_nodes = number_of_nodes
        self.original_capacity = original_capacity
        self.edges = [(x, x + 1, self.original_capacity) for x in range(1, self.number_of_nodes)]\
                                    + [(self.number_of_nodes, 1, self.original_capacity)]

    def dist(self, start_node: int, end_node: int):
        return min((start_node - end_node) % self.number_of_nodes,
                   (end_node - start_node) % self.number_of_nodes)


class GraphEdgesFactory:
    def __init__(self, number_of_nodes: int = 32, distance_threshold: int = 1, original_capacity: int = 1,
                 alpha = 1):

        self.physical_graph = PhysicalGraph(number_of_nodes, original_capacity)
        self.alpha = alpha
        self.number_of_nodes = number_of_nodes
        self.distance_threshold = distance_threshold
        self.original_capacity = original_capacity

    def shift_by_index(self, start: int, displacement: int):
        end = (start + displacement - 1) % self.number_of_nodes + 1
        return end

    def get_choice_probability(self, start_node: int, end_node: int):

        # Alternatively take out the end_node as well from the summing term
        beta_u = sum([1/(self.physical_graph.dist(start_node, x) ** self.alpha)
                      for x in range(1, self.number_of_nodes + 1) if x != start_node])

        physical_distance = self.physical_graph.dist(start_node, end_node)

        if 1 < physical_distance <= self.distance_threshold:
            prob = (1/beta_u)*(1/(physical_distance ** self.alpha))
            return prob
        else:
            return 0

    def power_law_link(self, arguments: tuple):

        start_node, long_link_distance = arguments

        i = 1

        # These are the nodes that can be selected while sampling
        possible_nodes = [self.shift_by_index(start_node, x) for x in range(-long_link_distance, long_link_distance+1)]
        power_law_distribution = [self.get_choice_probability(start_node, x) for x in possible_nodes]

        probabilities = np.array(power_law_distribution)
        probabilities /= probabilities.sum()
        np.random.seed()
        end_node = np.random.choice(possible_nodes, p=probabilities)

        # sample the end node for the long link
        # return numpy.random.choice(possible_nodes, p=power_law_distribution)
        # end_node = numpy.random.choice(possible_nodes, p=power_law_distribution)

        '''
        # Get the index of the end_node
        while random.random() > prob or (i % self.number_of_nodes) == 0:
            i += 1
            end_node = self.shift_by_index(start_node, i)
            prob = self.get_choice_probability(start_node, end_node)
        '''
        return end_node

    def deterministic_link(self, arguments: tuple):
        start_node, long_link_distance = arguments
        return self.shift_by_index(start_node, long_link_distance)

    def is_clockwise_indexed(self, start: int, end: int):
        # Checking:
        # If we regard the displacement from the start node by the distance between the start and the end
        # do we arrive in the end node?
        return self.shift_by_index(start, self.physical_graph.dist(start, end)) == end

    '''
    Reduces multiple edges into the same tuple
    '''
    def reduce_edges(self, edge_list: list):
        local_dictionary = {}
        local_list = []
        for (start, end, capacity) in edge_list:

            if end == 33 or start == 33:
                print('asd')
            # Swap the start and end nodes, if not indexed in the same direction
            if not self.is_clockwise_indexed(start, end):
                start, end = end, start

            # Add them to the directionary for keeping track
            if (start, end) in local_dictionary:
                local_dictionary[(start, end)] += capacity
            else:
                local_dictionary[(start, end)] = capacity
        for edge, capacity in local_dictionary.items():
            local_list.append((edge[0], edge[1], capacity))
        return local_list

    def generate_graph_edges(self, create_virtual_link = deterministic_link):

        virtual_links = []

        # Initiating the edges of 1) type: 1->2, 3->4, ... 31->32
        long_link_distance = min(self.distance_threshold, 2)
        virtual_links += [(x, create_virtual_link(arguments = (x, long_link_distance,)),
                           self.original_capacity) for x in range(1, self.number_of_nodes + 1)
                        if x % 2 == 1 and x < self.number_of_nodes]

        # Initiating the edges of 2) type: 1->2, 5->6, ... 29->30 (making up for the edges present in graph3)
        # Number of nodes have 0 as remainder for modulo 4
        long_link_distance = min(self.distance_threshold, 4)
        virtual_links += [(x, create_virtual_link(tuple((x, long_link_distance))),
                           self.original_capacity) for x in range(1, self.number_of_nodes)
                                       if x % 4 == 1]

        # Add up the multiple links
        return self.reduce_edges(self.physical_graph.edges + virtual_links)

