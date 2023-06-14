import sys
import random
import networkx
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import cmath
import math
import argparse
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
    # parse args
    parser = argparse.ArgumentParser(description="generate scale free networks with modification", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("number_of_node", help="number of nodes", type=int)
    parser.add_argument("average_degree", help="average degree", type=int)
    parser.add_argument("--remove_hubs", nargs='+', help="number of largest hubs to remove, can be an array: --remove_hubs 1 2 4 8 16 32")

    args = parser.parse_args()
    config = vars(args)
    if len(sys.argv) < 3:
        print("use: python ./nx_generate_scale_free_network.py {number_of_node} {average_degree} {other_args}*")
        exit(1)
    n = int(config['number_of_node'])
    average_degree = int(config['average_degree'])
    remove_hubs = config["remove_hubs"]

    total_edges = n * average_degree
    d = (n**2) - (4*total_edges)    # calculate the discriminant, d = (b**2) - (4*a*c)
    m = (n - cmath.sqrt(d))/2       # use the smaller root, sol1 = (-b-cmath.sqrt(d))/(2*a)
    m = m.real
    m_floor = math.floor(m)
    m_ceil = math.ceil(m)
    prob = m - m_floor

    G: nx.Graph = nx.dual_barabasi_albert_graph(n=n, m1=m_floor, m2=m_ceil, p=1-prob, seed=np.random, initial_graph=None)

    # shuffle nodes
    node_list = list(range(0, n))
    node_list_mapping = list(range(0, n))
    random.shuffle(node_list_mapping)
    mapping = {}
    for i in range(0, n):
        mapping[node_list[i]] = node_list_mapping[i]
    G = nx.relabel_nodes(G, mapping)

    if remove_hubs:
        for remove_hub_count in remove_hubs:
            G_temp = copy.deepcopy(G)
            node_degree_sorted = sorted(G_temp.degree, key=lambda x: x[1], reverse=True)
            for i in range(0, int(remove_hub_count)):
                G_temp.remove_node(node_degree_sorted[i][0])

            name = "n" + str(n) + "." + "avg" + str(round(m)) + "." + "remove_hub" + str(remove_hub_count)
            generate_topology_file(G_temp, name)
            save_network_info(G_temp, name)
    else:
        name = "n" + str(n) + "." + "avg" + str(round(m))
        generate_topology_file(G, name)
        save_network_info(G, name)



