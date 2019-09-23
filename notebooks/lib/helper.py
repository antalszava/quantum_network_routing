import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../lib")

import random
import scipy.stats as ss
import routing_simulation

"""
Applies func to each element of tup and returns a new tuple.

>>> a = (1, 2, 3, 4)
>>> func = lambda x: x * x
>>> map_tuple(func, a)
(1, 4, 9, 16)
"""


def map_tuple(func, tup):
    new_tuple = ()
    for itup in tup:
        new_tuple += (func(itup),)
    return new_tuple


"""
Applies func to each element of tup and returns a new tuple.

>>> a = (1, 2, 3, 4)
>>> func = lambda x: x * x
>>> map_tuple(func, a)
(1, 4, 9, 16)
"""


def map_tuple_gen(func, tup):
    return tuple(func(itup) for itup in tup)


'''
Add or update a key-value pair to the dictionary
'''


def add_tuple_to_dictionary(dictionary: dict, tup: tuple):
    k, v = tup
    if k in dictionary:
        dictionary[k] += v
    else:
        dictionary[k] = v
    return None


'''
Add or create elements of a dictionary to another dictionary
'''


def add_dictionary_to_dictionary(dictionary: dict, other_dictionary: dict):
    for k,v in other_dictionary.items():
        if k in dictionary:
            dictionary[k] += v
        else:
            dictionary[k] = v
    return None


def generate_random_source_destination(number_of_nodes: int) -> tuple:
    """
    Generates a random source and destination pair based on the number of nodes specified.

    Parameters
    ----------
    number_of_nodes : int
        Integer specifying the number of nodes in the graph.

    Returns
    -----
        Tuple containing the source and destination
    """
    random.seed()
    source = random.randint(1, number_of_nodes)
    dest = random.randint(1, number_of_nodes)
    while source == dest:
        dest = random.randint(1, number_of_nodes)
    return source, dest


def generate_random_pairs(number_of_pairs: int, number_of_nodes: int) -> list:
    """
    Generates a certain number of random source-destination pairs.

    Parameters
    ----------
    number_of_pairs : int
        Integer specifying the number of source-destination pairs to be generated.

    number_of_nodes : int
        Number of nodes used to generate random pairs.
    Returns
    -----
        List of tuples containing the source and destination nodes
    """
    result = []
    number_of_nodes = number_of_nodes

    for x in range(number_of_pairs):
        result += [generate_random_source_destination(number_of_nodes)]
    return result


def compute_mean_with_confidence(data: list, confidence: float = 0.95):

    # Calculating confidence interval values of the result
    n = len(data)
    std_err = ss.sem(data)
    degrees_of_freedom = n - 1
    h = std_err * ss.t.ppf((1 + confidence) / 2, degrees_of_freedom)
    return h


def extract_argument(argument_dictionary: dict, key: str, default_value):
    """
    Helper function that extracts an argument from a dictionary of arguments.
    Returns a default value if the key is not in the dictionary.

    Parameters
    ----------
    argument_dictionary: dict
        Dictionary of arguments to be used in the simulation.

    key: str
        The key whose value is to be extracted from the dictionary of arguments.

    default_value
        Default value
    """
    if key not in argument_dictionary.keys():
        extracted_argument=default_value
    else:
        extracted_argument=argument_dictionary[key]
    return extracted_argument