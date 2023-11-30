import time

import networkx as nx
import data_process_lib
import math
import matplotlib
import matplotlib.pyplot as plt
import itertools

config_file_path = 'simulator_config.json'
number_of_community = 16


def most_valuable_edge(G):
    t = time.time()
    betweenness = nx.edge_betweenness_centrality(G)
    # betweenness = nx.edge_load_centrality()
    print(f"elapsed: {time.time() - t}")
    return max(betweenness, key=betweenness.get)


if __name__ == "__main__":
    G = data_process_lib.load_graph_from_simulation_config(config_file_path)

    node_size = data_process_lib.calculate_node_size_for_drawing(G)
    layout = data_process_lib.try_load_data("layout.pkl")
    if layout is None:
        layout = nx.nx_agraph.graphviz_layout(G)
        data_process_lib.save_data(layout, "layout.pkl")

    # community
    node_color_dict = {}
    node_labels = {}
    comp = nx.community.girvan_newman(G, most_valuable_edge=most_valuable_edge)
    limited = itertools.takewhile(lambda c: len(c) <= number_of_community, comp)
    final_communities = next(limited)
    for communities in limited:
        final_communities = tuple(sorted(c) for c in communities)
    for i, single_community in enumerate(final_communities):
        print(f"community {i} -- {single_community}")
    for i, community in enumerate(final_communities):
        for single_node in community:
            node_color_dict[int(single_node)] = i
            node_labels[single_node] = f"{single_node}"

    min_key = min(node_color_dict.keys())
    max_key = max(node_color_dict.keys())
    node_color = []
    for key in range(min_key, max_key + 1):
        if key in node_color_dict:
            node_color.append(node_color_dict[key])
        else:
            pass

    cmap = matplotlib.colormaps.get_cmap('hsv')
    normalize = matplotlib.colors.Normalize(vmin=min(node_color), vmax=max(node_color))
    node_color_draw = [cmap(normalize(c)) for c in node_color]

    fig = plt.figure(frameon=False)
    fig.set_size_inches(12, 12)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    nx.draw(G, node_color=node_color_draw, with_labels=True, pos=layout, font_color='k', labels=node_labels, alpha=0.7, linewidths=0.1, width=0.1, font_size=8, node_size=node_size)

    fig.savefig("community.pdf", pad_inches=0)
    plt.close(fig)

