import matplotlib.pyplot as plt
import time
import os
import routing_simulation
import numpy


def plot_specific_measure(results: list, algo_name, topology_names: list, measures: list,  measure_index: int,
                          plot_type= 'approach', defined_color = True):
    timestr = time.strftime("%y_%m_%d__%H_%M")

    if defined_color:
        colors = ['red', 'green', 'blue', 'purple', 'darkorange']
    else:
        colors = numpy.random.rand(len(results), )

    for index in range(len(results)):
        plt.plot(results[index][measure_index], color=colors[index], label=topology_names[index])

    plt.grid(color='b', linestyle='-', linewidth=0.1)
    plt.xlabel('Number of SD pairs')
    plt.ylabel(measures[measure_index])
    if plot_type is 'approach':
        plt.title('Approach: ' + algo_name)
    elif plot_type is 'model':
        plt.title('Topology: ' + algo_name)
    plt.legend(loc=0)

    # Create the logs directory, if it is non-existant yet
    directory = 'plots'
    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig('./' + directory + '/' + algo_name + '_' + str(routing_simulation.Settings().number_of_samples) + '_' +
                measures[measure_index] + '_' + timestr +'.png', bbox_inches='tight')

    plt.show()


def plot_results(results: list, title: str, topology_names=None,
                                           measure_names=None, plot_type='algo'):

    if topology_names is None:
        # topology_names = ['on-demand', 'dth=1', 'dth=2', 'dth=4']
        topology_names = ['dth=1', 'dth=2', 'dth=4', 'dth=8', 'dth=16']
    if measure_names is None:
        measure_names = ['Average waiting times:', 'Number of available links:',
                         'Number of available edges:', 'Average distances:']

    for measure_index in range(len(measure_names)):
        plot_specific_measure(results, title, topology_names, measure_names, measure_index, plot_type)