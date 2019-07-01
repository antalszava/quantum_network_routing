
parse_from_log.py
'''
Returns the dumped data in the following format:

Return values:
parsed_topology_names: the topology names that were contained in the file (such as graph1)
parsed_measures: 
processed_results: results parsed from the log file and processed, so that they are contained as a list of tuples,
                   with as many elements as topologies,
                   each tuple containing the specific measures (e.g. AVG waiting time, distances, etc.)
'''
def parse_from_log(log_file_path: str, filename: str)

--------------------------------------
def plot_specific_measure(results: list, algo_name, topology_names: list, measures: list,  measure_index: int)



def plot_results(results:list, title: str, topology_names=None,
                                           measure_names=None):

    if measure_names is None:
        topology_names = ['graph0', 'graph1', 'graph2', 'graph3']
    if measure_names is None:
        measure_names = ['Average waiting times:', 'Number of available links:',
                         'Number of available edges:', 'Average distances:']
