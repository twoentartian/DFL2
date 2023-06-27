import sys
import networkx as nx
import numpy as np
import argparse

import nx_lib


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser(description="generate a herds networks", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("number_of_node", help="number of nodes", type=int)
    parser.add_argument("degree", help="degree for each node", type=int)
    parser.add_argument("number_of_herds", help="number of herds", type=int)
    parser.add_argument("degree_of_herds", help="node degree within the a herd", type=int)

    args = parser.parse_args()
    config = vars(args)
    if len(sys.argv) < 5:
        print("use: python ./nx_generate_herds_network.py {number_of_node} {degree} {number_of_herds} {degree_of_herds}")
        exit(1)
    n = int(config['number_of_node'])
    degree = int(config['degree'])
    number_of_herds = int(config['number_of_herds'])
    degree_of_herds = int(config['degree_of_herds'])

    print("number of nodes: " + str(n))
    print("degree: " + str(degree))
    print("number_of_herds: " + str(degree))
    print("degree_of_herds: " + str(degree_of_herds))

    degrees = [degree] * n
    G_base = nx.random_degree_sequence_graph(degrees, seed=np.random)

    for i in range(number_of_herds):
        node_name_shift = n // number_of_herds
        herd_size = n // number_of_herds
        degrees_herd = [degree_of_herds] * herd_size
        G_herd = nx.random_degree_sequence_graph(degrees_herd, seed=np.random)
        mapping = {}
        for i in range(0, len(G_herd.nodes)):
            mapping[i] = node_name_shift * i
        G_herd = nx.relabel_nodes(G_herd, mapping)
        G_base = nx_lib.combine_two_networks(G_base, G_herd)

    name = "n" + str(n) + "." + "degree" + str(degree) + "." + "herd" + str(number_of_herds) + "-" + str(degree_of_herds)
    nx_lib.generate_topology_file(G_base, name)
    nx_lib.save_network_info(G_base, name)
