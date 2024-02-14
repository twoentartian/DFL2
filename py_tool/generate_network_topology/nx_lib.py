import os.path
import random
import networkx as nx
import re
import matplotlib.pyplot as plt
import numpy as np
import json


def generate_topology_file(graph: nx.Graph, save_file_name: str):
    f = open(save_file_name + ".data", "w")
    for node, neighbor_dict in graph.adjacency():
        for single_neighbor in neighbor_dict:
            f.write(str(node) + " " + str(single_neighbor) + "\n")
    f.close()


def save_network_info(graph: nx.Graph, save_file_name: str, enable_topology: bool = False, only_mainland_in_topology=True):
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
        if only_mainland_in_topology:
            Gcc = graph.subgraph(sorted(nx.connected_components(graph), key=len, reverse=True)[0])
        else:
            Gcc = graph
        pos = nx.spring_layout(Gcc, seed=10396953)
        nx.draw_networkx_nodes(Gcc, pos, ax=ax0, node_size=20)
        nx.draw_networkx_edges(Gcc, pos, ax=ax0, alpha=0.4)
        labels = {}
        for node in Gcc.nodes():
            labels[node] = node
        nx.draw_networkx_labels(Gcc, pos, labels, ax=ax0, font_size=3, font_color='r')

        ax0.set_title("Network G")
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


def get_graph_from_dfl_simulation_config(config_file_path):
    config_file = ()
    if os.path.isfile(config_file_path):
        config_file = open(config_file_path)
    elif os.path.isdir(config_file_path):
        config_file = open(os.path.join(config_file_path, 'simulator_config.json'))
    config_file_content = config_file.read()
    config_file_json = json.loads(config_file_content)
    topology = config_file_json['node_topology']
    G = nx.Graph()
    for singleItem in topology:
        un_uir_link = singleItem.split('--')
        if len(un_uir_link) != 1:
            G.add_edge(un_uir_link[0], un_uir_link[1])

        dir_link = singleItem.split('->')
        if len(dir_link) != 1:
            G.add_edge(dir_link[0], dir_link[1])
    return G


def combine_two_networks(G0: nx.Graph, G1: nx.Graph) -> nx.Graph:
    output: nx.Graph = nx.Graph()

    all_nodes = []
    for node in G0.nodes:
        all_nodes.append(node)
    for node in G1.nodes:
        all_nodes.append(node)
    output.add_nodes_from(all_nodes)

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


def generate_star_network(n: int, star_count, shuffle: bool = True) -> nx.Graph:
    star_list = list(range(n))
    if shuffle:
        random.shuffle(star_list)
    star_list = star_list[0:star_count]

    G_temp: nx.Graph = nx.Graph()
    G_temp.add_nodes_from(range(n))
    for star in star_list:
        for node_index in range(n):
            if star != node_index:
                G_temp.add_edge(star, node_index)
    return G_temp