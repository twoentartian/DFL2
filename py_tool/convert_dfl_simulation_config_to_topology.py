import json
import networkx as nx
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


if __name__ == "__main__":
    config_file = open('./simulator_config.json')
    config_file_content = config_file.read()
    config_file_json = json.loads(config_file_content)
    topology = config_file_json['node_topology']
    G = nx.Graph()
    for singleItem in topology:
        unDirLink = singleItem.split('--')
        if len(unDirLink) != 1:
            G.add_edge(unDirLink[0], unDirLink[1])

        dirLink = singleItem.split('->')
        if len(dirLink) != 1:
            G.add_edge(dirLink[0], dirLink[1])
    name = "from_config"
    generate_topology_file(G, name)
    save_network_info(G, name)
