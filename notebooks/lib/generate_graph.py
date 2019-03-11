'''
Reduces multiple edges into the same tuple
'''
def reduce_edges(edge_list: list):
    local_dictionary = {}
    local_list = []
    for (start, end, capacity) in edge_list:
        if (start, end) in local_dictionary:
            local_dictionary[(start, end)] += capacity
        else:
            local_dictionary[(start, end)] = capacity
    for edge, capacity in local_dictionary.items():
        local_list.append((edge[0], edge[1], capacity))
    return local_list


def generate_graphs(number_of_nodes: int = 32, original_capacity:int = 1):
    # Initiating the edges of 0) type: 1->2, 2->3, ... 31->32, 32->1 (circular graph)
    graph0_edges = [(x, x + 1, 0) for x in range(1, number_of_nodes)] + [(number_of_nodes, 1, 0)]

    # Iniating the edges of 0) type: 1->2, 2->3, ... 31->32, 32->1 (circular graph)
    graph1_edges = [(x, x + 1, original_capacity) for x in range(1, number_of_nodes)] + [
        (number_of_nodes, 1, original_capacity)]

    # Iniating the edges of 1) type: 1->2, 3->4, ... 31->32
    # Odd number of nodes
    graph1_edges = graph1_edges + [(x, x + 1, original_capacity) for x in range(1, number_of_nodes + 1)
                                   if x % 2 == 1 and x < number_of_nodes and number_of_nodes % 2 == 1]

    # Even number of nodes
    graph1_edges = graph1_edges + [(x, (x + 1) % (number_of_nodes + 1), original_capacity) for x in
                                   range(1, number_of_nodes + 1) if x % 2 == 1 and x < number_of_nodes
                                   and number_of_nodes % 2 == 0]

    # Iniating the edges of 2) type: 1->2, 5->6, ... 29->30 (making up for the edges present in graph3)
    # Number of nodes have 0 as remainder for modulo 4
    graph1_edges = graph1_edges + [(x, x + 1, original_capacity) for x in range(1, number_of_nodes)
                                   if x % 4 == 1 and (x + 1) <= number_of_nodes] + [
                       (x, 1, original_capacity) for x in range(number_of_nodes - 1, number_of_nodes)
                       if x % 4 == 1 and (x + 1) > number_of_nodes]

    # Add up the multiple links:
    graph1_edges = reduce_edges(graph1_edges)

    # Iniating the edges of 0) type: 1->2, 2->3, ... 31->32, 32->1 (circular graph)
    graph2_edges = [(x, x + 1, original_capacity) for x in range(1, number_of_nodes)] + [
        (number_of_nodes, 1, original_capacity)]

    # Iniating the edges of 1) type: 1->3, 3->5, ... 31->1
    # Odd number of nodes
    graph2_edges = graph2_edges + [(x, x + 2, original_capacity) for x in range(1, number_of_nodes + 1)
                                   if x % 2 == 1 and x < number_of_nodes and number_of_nodes % 2 == 1]

    # Even number of nodes
    graph2_edges = graph2_edges + [(x, (x + 2) % number_of_nodes, original_capacity) for x in
                                   range(1, number_of_nodes + 1) if x % 2 == 1 and x < number_of_nodes
                                   and number_of_nodes % 2 == 0]

    # Adding additional long links to equal out the missing links compared to graph3
    # Iniating the edges of type: 1->3, 5->7, ... 29->31
    # Number of nodes have 0 as remainder for modulo 4
    graph2_edges = graph2_edges + [(x, x + 2, original_capacity) for x in range(1, number_of_nodes)
                                   if x % 4 == 1 and x + 2 <= number_of_nodes] + [
                       (x, 2 - (number_of_nodes - x), original_capacity) for x in range(1, number_of_nodes)
                       if x % 4 == 1 and (x + 2) > number_of_nodes]

    graph2_edges = reduce_edges(graph2_edges)

    # Initiating the edges of 0) type: 1->2, 2->3, ... 31->32, 32->1 (circular graph)
    graph3_edges = [(x, x + 1, original_capacity) for x in range(1, number_of_nodes)] + [
        (number_of_nodes, 1, original_capacity)]

    # Initiating the edges of 1) type: 1->3, 3->5, ... 31->1
    # Even number of nodes
    graph3_edges = graph3_edges + [(x, (x + 2) % number_of_nodes, original_capacity) for x in
                                   range(1, number_of_nodes + 1)
                                   if x % 2 == 1 and x < number_of_nodes and number_of_nodes % 2 == 0]
    # Odd number of nodes
    graph3_edges = graph3_edges + [(x, x + 2, original_capacity) for x in range(1, number_of_nodes + 1)
                                   if x % 2 == 1 and x < number_of_nodes and number_of_nodes % 2 == 1]

    # Initiating the edges of 2) type: 1->5, 5->9, ... 29->1
    # Number of nodes have 1 as remainder for modulo 4 (with indices smaller than or equal to the number of nodes)
    graph3_edges = graph3_edges + [(x, x + 4, original_capacity) for x in range(1, number_of_nodes)
                                   if x % 4 == 1 and x + 4 <= number_of_nodes]

    # Add the last link as well, that connects the greatest node with an index of 1 as remainder modulo 4 in distance
    graph3_edges = graph3_edges + [(x, 4 - (number_of_nodes - x), original_capacity) for x in range(1, number_of_nodes)
                                   if x % 4 == 1 and (x + 4) > number_of_nodes]

    return graph0_edges, graph1_edges, graph2_edges, graph3_edges