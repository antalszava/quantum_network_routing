import logging
import importlib
import time
import os
import routing_simulation


simulation_measures = ['Average waiting times:', 'Number of available links:',
                      'Number of available edges:', 'Average distances:']

directory = 'logs'

log = logging.getLogger()  # root logger

def write_results_to_file(results: dict, elapsed_time: int):

    initialize_logging()

    create_logs_directory()
    for algorithm in results.keys():
        algorithm_results(results[algorithm], algorithm, '', elapsed_time)

def algorithm_results(algorithm_results: list, algorithm: str, approach: str, elapsed_time: int):
    timestr = time.strftime("%y_%m_%d__%H_%M")

    details_in_name = [algorithm, str(routing_simulation.SimulationSettings().number_of_samples),
                       approach, timestr, '.log']
    filename = '_'.join(details_in_name)
    filepath = os.path.join(directory, filename)

    manage_handlers(filepath)

    algorithm_and_approach_info(algorithm, approach)

    dump_raw_results(algorithm_results)
    dump_elapsed_time(elapsed_time)

def manage_handlers(filepath):
    """Removes old handlers and adds a new one.

    Args:
        fileh (logging.FileHandler): the filehandler to be used
    """
    fileh = logging.FileHandler(filepath, 'a')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fileh.setFormatter(formatter)

    for hdlr in log.handlers[:]:  # remove all old handlers
        log.removeHandler(hdlr)
    log.addHandler(fileh)  # set the new handler

def initialize_logging():
    """Initialize the logging by reloading and setting the basic configuration.
    """
    importlib.reload(logging)
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')


def algorithm_and_approach_info(algorithm_name, approach):
    standard_message = 'Logging the simulation results of the ' + algorithm_name + ' algorithm'

    # Names of simulation measures
    if '' == approach:
        log.debug(standard_message + '.')
    else:
        log.debug(standard_message + '(using the ' + approach + '.')
    log.debug('The simulation measures are as follows: ')
    log.debug(simulation_measures)

def create_logs_directory():
    """Create the logs directory, if it is non-existant yet.
    """
    os.makedirs(directory, exist_ok=True)

def dump_raw_results(algorithm_results: list):
    """Log all the results of the simulation.

    Args:
        simulation_results (list): the simulations results to process
    """
    log.debug('Detailed simulation results based on the graph measured: ')
    for graph_index in range(len(algorithm_results)):
        log.debug('graph' + str(graph_index) + ':')

        # Log the stores containing the results
        dump_graph_results(algorithm_results[graph_index])

def dump_graph_results(graph_results: list):
    """Log the results a specific graph in the simulation.

    Args:
        graph_results (list): the graph results to process
    """
    for store_index in range(len(graph_results)):
        log.debug('-------------------------------------------------------------')
        log.debug(simulation_measures[store_index])
        log.debug('-------------------------------------------------------------')
        log.debug(graph_results[store_index])

def dump_elapsed_time(elapsed_time):
    """Log the results the elapsed time.

    Args:
        elapsed_time (int): the time that has elapsed since the start of the
            simulation
    """
    log.debug('The elapsed time was: ' + str(elapsed_time))
