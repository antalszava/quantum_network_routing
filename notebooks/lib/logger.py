import logging
import importlib
import time
import os
import routing_simulation

def log_graph_results(log: logging.Logger, graph_results: list, simulation_measures: list):
    for store_index in range(len(graph_results)):
        log.debug('-------------------------------------------------------------')
        log.debug(simulation_measures[store_index])
        log.debug('-------------------------------------------------------------')
        log.debug(graph_results[store_index])


def write_results_to_file(simulation_results: list, algorithm_name: str, approach: str, elapsed_time: int):
    importlib.reload(logging)
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')
    timestr = time.strftime("%y_%m_%d__%H_%M")

    # Create the logs directory, if it is non-existant yet
    directory = 'logs'
    if not os.path.exists(directory):
        os.makedirs(directory)

    fileh = logging.FileHandler('./' + directory + '/' + algorithm_name + '_' +
                                str(routing_simulation.Settings().number_of_samples) +
                                '_' + approach + '_' + timestr + '.log', 'a')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fileh.setFormatter(formatter)

    log = logging.getLogger()  # root logger
    for hdlr in log.handlers[:]:  # remove all old handlers
        log.removeHandler(hdlr)
    log.addHandler(fileh)  # set the new handler

    # Names of simulation measures
    simulation_measures = ['Average waiting times:', 'Number of available links:',
                          'Number of available edges:', 'Average distances:']

    if '' == approach:
        log.debug('Logging the simulation results of the ' + algorithm_name + ' algorithm.')
    else:
        log.debug(
            'Logging the simulation results of the ' + algorithm_name + ' algorithm (using the ' + approach + '.')
    log.debug('The simulation measures are as follows: ')
    log.debug(simulation_measures)
    log.debug('Detailed simulation results based on the graph measured: ')
    for graph_index in range(len(simulation_results)):
        log.debug('graph' + str(graph_index) + ':')

        # Log the stores containing the results
        log_graph_results(log, simulation_results[graph_index], simulation_measures)

    # Log the elapsed time
    log.debug('The elapsed time was: ' + str(elapsed_time))
