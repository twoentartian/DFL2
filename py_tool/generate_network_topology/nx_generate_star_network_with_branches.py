import random
import sys
import networkx as nx
import argparse

import nx_lib


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser(description="generate a star network with branches", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("number_of_branch", help="number of branches", type=int)
    parser.add_argument("depth_of_branch", nargs='+', help="depth of each branch", type=int)

    args = parser.parse_args()
    config = vars(args)
    if len(sys.argv) < 3:
        print("use: python ./nx_generate_star_network_with_branches.py {number_of_branch} {depth_of_branch}")
        exit(1)
    number_of_branch = int(config['number_of_branch'])
    depth_of_branch = config['depth_of_branch']

    print("number_of_branch: " + str(number_of_branch))
    print("depth_of_branch: " + str(depth_of_branch))

    for current_depth in depth_of_branch:
        G_temp: nx.Graph = nx.Graph()
        G_temp.add_node(0)

        edge_list = []
        node_index = 1  # starts from 1
        for b in range(number_of_branch):
            previous_node = 0
            for d in range(current_depth):
                edge_list.append((previous_node, node_index))
                previous_node = node_index
                node_index = node_index + 1
        G_temp.add_edges_from(edge_list)

        name = f"star.branch{number_of_branch}.depth{current_depth}"
        nx_lib.generate_topology_file(G_temp, name)
        nx_lib.save_network_info(G_temp, name, True)
        print(f"done generating {name}")

