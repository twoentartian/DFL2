import sys
import networkx as nx
import numpy as np
import argparse

import nx_lib


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser(description="generate star networks", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("number_of_node", help="number of nodes", type=int)
    parser.add_argument("degree", help="degree for each node", type=int)

    args = parser.parse_args()
    config = vars(args)
    if len(sys.argv) < 3:
        print("use: python ./nx_generate_star_network.py {number_of_node} {number_of_star}")
        exit(1)
    n = int(config['number_of_node'])
    degree = int(config['degree'])

    print("number of nodes: " + str(n))
    print("degree: " + str(degree))

    degrees = [degree] * n
    G = nx.random_degree_sequence_graph(degrees, seed=np.random)

    name = "n" + str(n) + "." + "degree" + str(degree)
    nx_lib.generate_topology_file(G, name)
    nx_lib.save_network_info(G, name)
