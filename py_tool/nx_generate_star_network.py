import sys
import networkx as nx
import argparse

import nx_lib


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser(description="generate star networks", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("number_of_node", help="number of nodes", type=int)
    parser.add_argument("number_of_star", help="number of nodes", type=int)

    args = parser.parse_args()
    config = vars(args)
    if len(sys.argv) < 3:
        print("use: python ./nx_generate_star_network.py {number_of_node} {number_of_star}")
        exit(1)
    n = int(config['number_of_node'])
    number_of_star = int(config['number_of_star'])

    print("number of nodes: " + str(n))
    print("number of star: " + str(number_of_star))

    G: nx.Graph = nx.Graph()
    for star_index in range(number_of_star):
        G_temp: nx.Graph = nx.star_graph(n=n-1)
        G_temp = nx_lib.shuffle_node(G_temp)
        all_edges = []
        for edge in G_temp.edges:
            all_edges.append((edge[0], edge[1]))
        G.add_edges_from(all_edges)

    name = "n" + str(n) + "." + "star" + str(number_of_star)
    nx_lib.generate_topology_file(G, name)
    nx_lib.save_network_info(G, name)
