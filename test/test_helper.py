import unittest


import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../lib")
import helper

class TestHelper(unittest.TestCase):

    """
    Test adding exactly one element to empty dictionary
    """
    def test_no_key_yet_add_tuple_to_dictionary(self):
        local_dictionary = {}
        tuple_to_add = (1, 2)

        helper.add_tuple_to_dictionary(local_dictionary, tuple_to_add)
        self.assertEqual(len(local_dictionary), 1)
        self.assertIn(tuple_to_add[0], local_dictionary, "The specified key was not added to the dictionary.")
        self.assertIn(tuple_to_add[1], local_dictionary.values(), "The specified value was not added to the dictionary.")

    """
    Test adding exactly one element to empty dictionary
    """
    def test_update_value_add_tuple_to_dictionary(self):
        local_dictionary = {'a': 0}
        local_range = range(10000)

        for x in local_range:
            helper.add_tuple_to_dictionary(local_dictionary, ('a',x))
        self.assertEqual(local_dictionary['a'], sum([x for x in local_range]))

    """
    Test adding when key exists, but the value is not a number
    """

    def test_value_type_not_number_add_tuple_to_dictionary(self):
        local_dictionary = {'a': 'b'}
        tuple_to_add = ('a', 1)
        self.assertRaises(TypeError, helper.add_tuple_to_dictionary, (local_dictionary, tuple_to_add))


if __name__ == '__main__':
    unittest.main()
