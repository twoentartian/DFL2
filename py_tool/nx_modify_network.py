import argparse
import random

import networkx as nx
import copy
from pathlib import Path

import nx_lib

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="rewire the network edges", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input_file", help="the topology file path, draw topology if no optional arguments", type=str)
    parser.add_argument("--rewire_ratio", nargs='+', help="the ratio of rewiring edges, can be an array: --rewire_ratio 0.01 0.02 0.04 0.08 0.16 0.32")
    parser.add_argument("--rewire_count", nargs='+', help="the count of rewiring edges, can be an array: --rewire_count 1 10 100 1000")
    parser.add_argument("--remove_hubs", nargs='+', help="number of largest hubs to remove, can be an array: --remove_hubs 1 2 4 8 16 32")
    parser.add_argument("--remove_hubs_above", nargs='+', help="remove the hubs above the limit, can be an array: --remove_hubs_above 100 500 1000")
    parser.add_argument("--downgrade_hubs", nargs="+", help="reduce hubs' degree to n: --downgrade_hubs n 100 500 1000")

    args = parser.parse_args()
    config = vars(args)

    file_name = Path(config["input_file"]).stem
    # load the file
    G = nx_lib.load_graph(config["input_file"])

    # check arg
    count = 0
    if config["rewire_ratio"]:
        count += 1
    if config["rewire_count"]:
        count += 1
    if config["remove_hubs"]:
        count += 1
    if config["remove_hubs_above"]:
        count += 1
    if config["downgrade_hubs"]:
        count += 1
    if count > 1:
        print("please specify only 1 optional parameter per time")
        exit(1)

    if count == 0:
        nx_lib.save_network_info(G, file_name)

    # rewire_ratio
    if config["rewire_ratio"]:
        for rewire_ratio in config["rewire_ratio"]:
            G_temp = copy.deepcopy(G)
            total_edges_to_rewire = len(G_temp.edges) * float(rewire_ratio)
            nx.connected_double_edge_swap(G_temp, nswap=total_edges_to_rewire/2)
            name = file_name + "." + "rewire_ratio" + str(rewire_ratio)
            nx_lib.generate_topology_file(G_temp, name)
            nx_lib.save_network_info(G_temp, name)

    # rewire_count
    if config["rewire_count"]:
        for rewire_count in config["rewire_count"]:
            G_temp = copy.deepcopy(G)
            total_edges_to_rewire = int(rewire_count) * 2
            nx.connected_double_edge_swap(G_temp, nswap=total_edges_to_rewire/2)
            name = file_name + "." + "rewire_count" + str(rewire_count)
            nx_lib.generate_topology_file(G_temp, name)
            nx_lib.save_network_info(G_temp, name)

    # remove_hubs
    if config["remove_hubs"]:
        for downgrade_hub_count in config["remove_hubs"]:
            G_temp = copy.deepcopy(G)
            node_degree_sorted = sorted(G_temp.degree, key=lambda x: x[1], reverse=True)
            for i in range(0, int(downgrade_hub_count)):
                G_temp.remove_node(node_degree_sorted[i][0])

            name = file_name + "." + "remove_hub" + str(downgrade_hub_count)
            nx_lib.generate_topology_file(G_temp, name)
            nx_lib.save_network_info(G_temp, name)

    # remove_hubs_above
    if config["remove_hubs_above"]:
        for remove_hubs_above in config["remove_hubs_above"]:
            G_temp = copy.deepcopy(G)
            nodes_to_remove = []
            for node_degree in G_temp.degree:
                if node_degree[1] > int(remove_hubs_above):
                    nodes_to_remove.append(node_degree[0])
            G_temp.remove_nodes_from(nodes_to_remove)
            name = file_name + "." + "remove_hubs_above" + remove_hubs_above
            nx_lib.generate_topology_file(G_temp, name)
            nx_lib.save_network_info(G_temp, name)

    # downgrade_hubs
    if config["downgrade_hubs"]:
        downgrade_hubs_args = config["downgrade_hubs"]
        downgrade_to_k = downgrade_hubs_args[0]
        hub_count_list = downgrade_hubs_args[1:]
        for downgrade_hub_count in hub_count_list:
            G_temp = copy.deepcopy(G)
            node_degree_sorted = sorted(G_temp.degree, key=lambda x: x[1], reverse=True)
            for i in range(0, int(downgrade_hub_count)):
                to_be_removed_node_name = node_degree_sorted[i][0]
                neighbors = set(G.neighbors(to_be_removed_node_name))
                G_temp.remove_node(node_degree_sorted[i][0])
                if len(neighbors) % 2 == 0:
                    # 4 nodes connect to removed_node
                    reconnect_list = random.sample(list(neighbors), 4)
                    for i in reconnect_list:
                        G_temp.add_edge(to_be_removed_node_name, i)
                        neighbors.remove(i)
                else:
                    reconnect_list = random.sample(list(neighbors), 3)
                    for i in reconnect_list:
                        G_temp.add_edge(to_be_removed_node_name, i)
                        neighbors.remove(i)
                assert len(neighbors) % 2 == 0
                neighbors = list(neighbors)
                random.shuffle(neighbors)
                for i in range(0, len(neighbors), 2):
                    G_temp.add_edge(neighbors[i], neighbors[i+1])
            name = file_name + "." + "downgrade_hubs" + downgrade_hub_count
            nx_lib.generate_topology_file(G_temp, name)
            nx_lib.save_network_info(G_temp, name)