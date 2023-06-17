import sys
from pathlib import Path
import networkx as nx
import argparse

import nx_lib


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser(description="combine two networks", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("graph0", help="path of first network", type=str)
    parser.add_argument("graph1", help="path of second network", type=str)

    args = parser.parse_args()
    config = vars(args)
    if len(sys.argv) < 3:
        print("use: python ./nx_combine_two_networks.py {graph0} {graph1}")
        exit(1)

    print("network 0: " + config["graph0"])
    print("network 1: " + config["graph1"])

    G0 = nx_lib.load_graph(config["graph0"])
    G1 = nx_lib.load_graph(config["graph1"])
    G: nx.Graph = nx_lib.combine_two_networks(G0, G1)

    filename0 = Path(config["graph0"]).stem
    filename1 = Path(config["graph1"]).stem

    name = "combine_" + filename0 + "_" + filename1
    nx_lib.generate_topology_file(G, name)
    nx_lib.save_network_info(G, name)
