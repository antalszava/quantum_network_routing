import matplotlib.pyplot as plt
import time
import os
import routing_simulation

def plot_specific_measure(results: list, algo_name, topology_names: list, measures: list,  measure_index: int):
    timestr = time.strftime("%y_%m_%d__%H_%M")
    colors = ['red', 'green', 'blue', 'purple']
    for index in range(len(results)):
        plt.plot(results[index][measure_index],color=colors[index], label=topology_names[index])
    plt.grid(color='b', linestyle='-', linewidth=0.1)
    plt.xlabel('Number of SD pairs')
    plt.ylabel(measures[measure_index])
    plt.title('Algorithm:' + algo_name)
    plt.legend()

    # Create the logs directory, if it is non-existant yet
    directory = 'plots'
    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig('./' + directory + '/' + algo_name + '_' + str(routing_simulation.Settings().number_of_samples) + '_' + measures[measure_index] +
                '_' + timestr +'.png', bbox_inches='tight')

    plt.show()

def plot_results(results:list, title: str, topology_names=['graph0', 'graph1', 'graph2', 'graph3'],
                                           measure_names=['Average waiting times:', 'Number of available links:',
                          'Number of available edges:', 'Average distances:']):
    for measure_index in range(len(measure_names)):
        plot_specific_measure(results, title, topology_names, measure_names, measure_index)
    #plot_average_distances(results, title, topology_names)
    #plot_unavailable_virtual_links(results, title, topology_names)