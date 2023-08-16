import sys
from pathlib import Path
import networkx as nx
import argparse

import nx_lib


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="shift the network nodes name (100+shift_value(200) = 300)", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input_file", help="the topology file path", type=str)
    parser.add_argument("shift_value", help="the value of shifting", type=int)

    args = parser.parse_args()
    config = vars(args)

    shift_value = config["shift_value"]

    file_name = Path(config["input_file"]).stem
    # load the file
    G = nx_lib.load_graph(config["input_file"])

    mapping = {}
    for i in range(0, len(G.nodes)):
        mapping[i] = i + shift_value
    G = nx.relabel_nodes(G, mapping)

    output_file_name = file_name + ".shift_" + str(shift_value)
    nx_lib.generate_topology_file(G, output_file_name)
    nx_lib.save_network_info(G, output_file_name)
