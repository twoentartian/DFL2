import sys
from pathlib import Path
import networkx as nx
import argparse

import nx_lib


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser(description="combine multiple networks", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("networks", nargs='+', help="path of networks")

    args = parser.parse_args()
    config = vars(args)
    if len(sys.argv) < 3:
        print("use: python ./nx_combine_networks.py {graph0} {graph1} {graph2}...")
        exit(1)

    G = nx.Graph()
    filename = "combine"
    for network_index in range(len(config["networks"])):
        config_item = config["networks"][network_index]
        print("network " + str(network_index) + ": " + config["networks"][network_index])
        G_temp = nx_lib.load_graph(config_item)
        G = nx_lib.combine_two_networks(G, G_temp)
        filename = filename + "_" + Path(config_item).stem

    nx_lib.generate_topology_file(G, filename)
    nx_lib.save_network_info(G, filename)
