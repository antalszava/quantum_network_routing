# Monte Carlo simulations for quantum network routing

## General introduction
A quantum network is created by specifying the properties of a virtual graph (graph containing entangled links). An underlying physical graph of ring topology is assumed. Once the network has been created, a fixed number of random traffic is generated. The quantum network serves the traffic by provisioning entangled links for quantum teleportation. Once the entangled links have been used for this procedure, they are removed from the graph representation of the network. Once unavailable, these links need to be regenerated, creating latency for a new traffic. This library serves as a tool to carry out numerical simulations for estimating the average latency for a demand.

## Overivew of the source files in the library
* **routing_simulation.py:** the main file containing methods used for initializing simulations.
* **routing_algorithms.py:** implementation of the knowledge based approaches used during the simulations.
* **shortest_path.py:** implementation of the solution to the single-source shortest path problem in a graph with non-negative weights using Dijkstra's algorithm.
* **graph.py:** implementation of a graph class used in the library.
* **graph_edge_factory.py:** includes a class with which the edges for deterministic or random graphs can be produced.
* **helper.py:** supplementary helper functions.
* **plot.py:** supplementary functions easing the plotting procedure.

## Arguments to be specified for each of the simulations (see example notebooks)
* **algorithm:** the knowledge based approach we would like to simulate. Values can be the following function objects: initial_knowledge_init, global_knowledge_init or local_knowledge_algorithm.
* **graph_edges:** in case of a deterministically created graph, a list of edges contained in the graph. Each edge is a tuple of the following format: (start node, end node, capacity)
* **distance_threshold:** the value determining the maximum length of long links in the graph
* **propagation_radius:** the radius of propagation used for the local knowledge approach
* **link_prediction:** a boolean value determining whether or not the link prediction technique is to be used
* **exponential_scale:** a boolean value determining whether or not the rebuilding of virtual links scales exponentially with the physical distance (else it scales polynomially)

The example Jupyter notebooks contain simulation scenarios for each of the "knowledge based approaches".
