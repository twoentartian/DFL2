import argparse
import networkx
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import re
import os
import copy


def generate_topology_file(graph: networkx.Graph, save_file_name: str):
    f = open(save_file_name + ".data", "w")
    for node, neighbor_dict in G.adjacency():
        for single_neighbor in neighbor_dict:
            f.write(str(node) + " " + str(single_neighbor) + "\n")
    f.close()


def save_network_info(graph: networkx.Graph, save_file_name: str):
    degree_sequence = sorted((d for n, d in graph.degree()), reverse=True)
    fig = plt.figure("Degree of a random graph", figsize=(8, 8))
    axgrid = fig.add_gridspec(5, 4)
    ax0 = fig.add_subplot(axgrid[0:3, :])
    Gcc = graph.subgraph(sorted(nx.connected_components(graph), key=len, reverse=True)[0])
    pos = nx.spring_layout(Gcc, seed=10396953)
    nx.draw_networkx_nodes(Gcc, pos, ax=ax0, node_size=20)
    nx.draw_networkx_edges(Gcc, pos, ax=ax0, alpha=0.4)
    ax0.set_title("Scale-Free Network G")
    ax0.set_axis_off()
    ax1 = fig.add_subplot(axgrid[3:, :2])
    ax1.plot(degree_sequence, "b-", marker="o")
    ax1.set_title("Degree Distribution Plot ({size} nodes)".format(size=len(graph.nodes)))
    ax1.set_ylabel("Degree")
    ax1.set_xlabel("Node Count")
    ax2 = fig.add_subplot(axgrid[3:, 2:])
    ax2.bar(*np.unique(degree_sequence, return_counts=True))
    ax2.set_title("Degree histogram")
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("Node Count")
    fig.tight_layout()
    fig.savefig(save_file_name + ".pdf")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="rewire the network edges", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input_file", help="the topology file path", type=str)
    parser.add_argument("--rewire_ratio", nargs='+', help="the ratio of rewiring edges, can be an array: --rewire_ratio 0.01 0.02 0.04 0.08 0.16 0.32")
    parser.add_argument("--remove_hubs", nargs='+', help="number of largest hubs to remove, can be an array: --remove_hubs 1 2 4 8 16 32")
    parser.add_argument("--remove_hubs_above", nargs='+', help="remove the hubs above the limit, can be an array: --keep_hubs_below 100 500 1000")

    args = parser.parse_args()
    config = vars(args)

    # load the file
    G: nx.Graph = nx.Graph()
    file_name = os.path.basename(config["input_file"])
    topology_file = open(config["input_file"], 'r')
    lines = topology_file.readlines()
    edge_list = []
    for line in lines:
        matches = re.findall(r'(\d+)', line)
        assert (len(matches) == 2)
        edge_list.append((int(matches[0]), int(matches[1])))
    G.add_edges_from(edge_list)

    # check arg
    count = 0
    if config["rewire_ratio"]:
        count += 1
    if config["remove_hubs"]:
        count += 1
    if config["remove_hubs_above"]:
        count += 1
    if count > 1:
        print("please specify only 1 optional parameter per time")
        exit(1)

    # rewire_ratio
    if config["rewire_ratio"]:
        for rewire_ratio in config["rewire_ratio"]:
            G_temp = copy.deepcopy(G)
            total_edges_to_rewire = len(G_temp.edges) * float(config["rewire_ratio"])
            nx.connected_double_edge_swap(G_temp, nswap=total_edges_to_rewire/2)
            name = file_name + "." + "rewire_ratio" + str(rewire_ratio)
            generate_topology_file(G_temp, name)
            save_network_info(G_temp, name)

    # remove_hubs
    if config["remove_hubs"]:
        for remove_hub_count in config["remove_hubs"]:
            G_temp = copy.deepcopy(G)
            node_degree_sorted = sorted(G_temp.degree, key=lambda x: x[1], reverse=True)
            for i in range(0, int(remove_hub_count)):
                G_temp.remove_node(node_degree_sorted[i][0])

            name = file_name + "." + "remove_hub" + str(remove_hub_count)
            generate_topology_file(G_temp, name)
            save_network_info(G_temp, name)

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
            generate_topology_file(G_temp, name)
            save_network_info(G_temp, name)