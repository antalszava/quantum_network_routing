import matplotlib.pyplot as plt
import time
import os
import numpy
import tikzplotlib


def plot_specific_measure(results: list, algo_name, topology_names: list, measures: list,
                          measure_index: int, number_of_samples: int = 1000, plot_type='approach',
                          defined_color: bool = True, save_tikz: bool = True):
    """
    Plots the specific measures coming from the results.

    Parameters
    ----------
    results : list of tuple containing the results
        The graph in which we serve the demands according to the global knowledge approach.

    algo_name: str
        Name of the algorithms used.

    topology_names: list
        Name of the topologies in which the simulation took place.

    measures: list
        Name of the measures to be plotted.

    measure_index: int
        Index of the measure to be plotted.

    number_of_samples: int
        Number of samples used in the simulation.

    plot_type: str
        Type of the plot

    defined_color: bool
        Boolean determining whether or not the pre-defined colors are to be used.

    save_tikz: bool
        Value defining whether or not to save as a tikz file.
    """
    timestr = time.strftime("%y_%m_%d__%H_%M")

    if defined_color:
        colors = ['red', 'green', 'blue', 'purple', 'darkorange', 'magenta']
    else:
        colors = numpy.random.rand(len(results), )

    markers = ["^", "o", "*", "D", "p"]
    for index in range(len(results)):
        y = results[index][measure_index]
        plt.plot(range(1, len(y)+1), y, color=colors[index], label=topology_names[index], marker=markers[index],
                 markersize=4)

    plt.grid(color='b', linestyle='-', linewidth=0.1)
    plt.xlabel('Number of source-destination pairs')
    plt.ylabel(measures[measure_index])

    if plot_type is 'approach':
        plt.title('Approach: ' + algo_name)
    elif plot_type is 'model':
        plt.title('Topology: ' + algo_name)

    plt.legend(loc=0)

    # Create the logs directory, if it is non-existent yet
    directory = 'plots'
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not save_tikz:
        plt.savefig('./' + directory + '/' + algo_name + '_' + str(number_of_samples) + '_' +
                    measures[measure_index] + '_' + timestr + '.png', bbox_inches='tight')
        plt.show()
    else:
        tikzplotlib.save('./' + directory + '/' + algo_name + '_' + str(number_of_samples) + '_' +
                             measures[measure_index] + '_' + timestr + '.tex')
        plt.clf()


def plot_results(results: list, title: str, topology_names=None,
                 measure_names=None, plot_type='algo', save_tikz: bool = True):
    """
    Plot the simulation results.

    Parameters
    ----------
    results : list of tuple containing the results
        The graph in which we serve the demands according to the global knowledge approach.

    title: str
        Title of the simulation.

    topology_names: list
        Name of the topologies in which the simulation took place.

    measure_names: list
        Name of the measures to be plotted.

    plot_type: str
        Type of the plot

    save_tikz: bool
        Value defining whether or not to save as a tikz file.
    """
    if topology_names is None:
        topology_names = ['dth=1', 'dth=2', 'dth=4', 'dth=8', 'dth=16']
        # Alternative topology names:
        # topology_names = ['on-demand', 'dth=1', 'dth=2', 'dth=4']
    if measure_names is None:
        measure_names = ['Average waiting times:', 'Number of available links:',
                         'Number of available edges:', 'Average distances:']

    for measure_index in range(len(measure_names)):
        plot_specific_measure(results, title, topology_names, measure_names, measure_index,
                              plot_type=plot_type, save_tikz=save_tikz)
