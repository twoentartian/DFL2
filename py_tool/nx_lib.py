import random
import networkx as nx
import re
import matplotlib.pyplot as plt
import numpy as np


def generate_topology_file(graph: nx.Graph, save_file_name: str):
    f = open(save_file_name + ".data", "w")
    for node, neighbor_dict in graph.adjacency():
        for single_neighbor in neighbor_dict:
            f.write(str(node) + " " + str(single_neighbor) + "\n")
    f.close()


def save_network_info(graph: nx.Graph, save_file_name: str, enable_topology: bool = False):
    degree_sequence = sorted((d for n, d in graph.degree()), reverse=True)

    if enable_topology:
        fig = plt.figure("Degree of a random graph", figsize=(8, 8))
        axgrid = fig.add_gridspec(5, 4)
        ax0 = fig.add_subplot(axgrid[0:3, :])
        ax1 = fig.add_subplot(axgrid[3:, :2])
        ax2 = fig.add_subplot(axgrid[3:, 2:])
    else:
        fig = plt.figure("Degree of a random graph", figsize=(8, 4))
        axgrid = fig.add_gridspec(2, 4)
        ax1 = fig.add_subplot(axgrid[:, :2])
        ax2 = fig.add_subplot(axgrid[:, 2:])

    if enable_topology:
        Gcc = graph.subgraph(sorted(nx.connected_components(graph), key=len, reverse=True)[0])
        pos = nx.spring_layout(Gcc, seed=10396953)
        nx.draw_networkx_nodes(Gcc, pos, ax=ax0, node_size=20)
        nx.draw_networkx_edges(Gcc, pos, ax=ax0, alpha=0.4)
        ax0.set_title("Scale-Free Network G")
        ax0.set_axis_off()

    ax1.plot(degree_sequence, "b-", marker="o")
    ax1.set_title("Degree Distribution Plot ({size} nodes)".format(size=len(graph.nodes)))
    ax1.set_ylabel("Degree")
    ax1.set_xlabel("Node Count")

    ax2.bar(*np.unique(degree_sequence, return_counts=True))
    ax2.set_title("Degree histogram")
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("Node Count")

    fig.tight_layout()
    fig.savefig(save_file_name + ".pdf")
    plt.close()


def load_graph(file_path: str) -> nx.Graph:
    G: nx.Graph = nx.Graph()
    topology_file = open(file_path, 'r')
    lines = topology_file.readlines()
    edge_list = []
    for line in lines:
        matches = re.findall(r'(\d+)', line)
        if len(matches) != 2:
            continue
        edge_list.append((int(matches[0]), int(matches[1])))
    G.add_edges_from(edge_list)
    return G


def combine_two_networks(G0: nx.Graph, G1: nx.Graph) -> nx.Graph:
    output: nx.Graph = nx.Graph()
    all_edges = []
    for edge in G0.edges:
        all_edges.append((edge[0], edge[1]))
    for edge in G1.edges:
        all_edges.append((edge[0], edge[1]))
    output.add_edges_from(all_edges)
    return output


def shuffle_node(G: nx.Graph) -> nx.Graph:
    n = len(G.nodes)
    node_list = list(range(0, n))
    node_list_mapping = list(range(0, n))
    random.shuffle(node_list_mapping)
    mapping = {}
    for i in range(0, n):
        mapping[node_list[i]] = node_list_mapping[i]
    G = nx.relabel_nodes(G, mapping)
    return G


def generate_random_network(n: int, degree: int) -> nx.Graph:
    degrees = [degree] * n
    G = nx.random_degree_sequence_graph(degrees, seed=np.random)
    return G


def generate_star_network(n: int, star_count) -> nx.Graph:
    star_list = list(range(n))
    random.shuffle(star_list)
    star_list = star_list[0:star_count]

    G_temp: nx.Graph = nx.Graph()
    G_temp.add_nodes_from(range(n))
    for star in star_list:
        for node_index in range(n):
            if star != node_index:
                G_temp.add_edge(star, node_index)
    return G_temp