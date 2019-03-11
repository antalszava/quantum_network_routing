'''
Applies func to each element of tup and returns a new tuple.

>>> a = (1, 2, 3, 4)
>>> func = lambda x: x * x
>>> map_tuple(func, a)
(1, 4, 9, 16)
'''
def map_tuple(func, tup):
    new_tuple = ()
    for itup in tup:
        new_tuple += (func(itup),)
    return new_tuple

'''
Applies func to each element of tup and returns a new tuple.

>>> a = (1, 2, 3, 4)
>>> func = lambda x: x * x
>>> map_tuple(func, a)
(1, 4, 9, 16)
'''
def map_tuple_gen(func, tup):
    return tuple(func(itup) for itup in tup)

'''
Returns the mean of an iterable (e.g. list of numbers)
'''
def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)