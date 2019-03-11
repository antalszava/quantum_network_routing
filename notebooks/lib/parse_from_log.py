import re
import ast

'''
Returns the dumped data in the following format:

Return values:
parsed_topology_names: the topology names that were contained in the file (such as graph1)
parsed_measures: 
processed_results: results parsed from the log file and processed, so that they are contained as a list of tuples,
                   with as many elements as topologies,
                   each tuple containing the specific measures (e.g. AVG waiting time, distances, etc.)
'''
def parse_from_log(log_file_path: str, filename: str):
    
    # Basic regex variables
    timestamp_regex = '[0-9:\- ]'
    alphanumeric_regex = '[0-9a-zA-Z]*'
    floating_list_regex = '[0-9., ]'
    debugging_regex = ' - root - DEBUG - '
    delimeter_regex = '-------'
    
    # Regex for the file
    measures_regex = '\[\'[a-zA-Z.,: \']*\'\]'
    #'(?P<topology_name>)' +
    topology_names_regex = '\- ' +  '(?P<topology_name>' + alphanumeric_regex + ')\:' + '\n'
    results_regex = '\[' + floating_list_regex + '*\]'

    # Stores for the found strings
    parsed_measures = []
    parsed_topology_names = []
    parsed_results = []
    
    with open(log_file_path + "/" + filename, "r") as log_file:
        for line in log_file:
            
            for match in re.finditer(measures_regex, line, re.S):
                matched = match.group()
                parsed_measures.append(matched)
            
            for match in re.finditer(results_regex, line, re.S):
                matched = match.group()
                parsed_results.append(matched)
                
            for match in re.finditer(topology_names_regex, line, re.S):
                matched = match.group('topology_name')
                parsed_topology_names.append(matched)
    
    # Evaluate the string of list as a list
    parsed_results = [ast.literal_eval(x) for x in parsed_results]
    parsed_measures = [ast.literal_eval(x) for x in parsed_measures][0]
    
    count_tuples = 0
    temp_tuple = ()
        
    processed_results = []
    
    # Create len(parsed_topology_names) many len(parsed_measures)-tuples
    for result_index in range(len(parsed_results)):
        temp_tuple += (parsed_results[result_index],)
        
        # See when we have processed the specific measures for a given topology already
        if len(temp_tuple) % len(parsed_measures) == 0:
            processed_results.append(temp_tuple)
            temp_tuple = ()
            
    return parsed_topology_names, parsed_measures, processed_results