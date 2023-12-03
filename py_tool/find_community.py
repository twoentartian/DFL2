import os.path
import time

import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import itertools
import importlib.util

import data_process_lib

config_file_path = 'simulator_config.json'
number_of_community = 128
output_community_folder = 'community'

spam_spec = importlib.util.find_spec("cugraph")
found_cu_graph = spam_spec is not None
if found_cu_graph:
    print("use cugraph backend")
else:
    print("use networkx backend")



def most_valuable_edge(G):
    t = time.time()

    # check cu-graph
    betweenness = 0
    if found_cu_graph:
        import cugraph as cg
        betweenness = cg.edge_betweenness_centrality(G)
    else:
        betweenness = nx.edge_betweenness_centrality(G)
    print(f"elapsed: {time.time() - t}")
    return max(betweenness, key=betweenness.get)


if __name__ == "__main__":
    G = data_process_lib.load_graph_from_simulation_config(config_file_path)

    if not os.path.exists(output_community_folder):
        os.mkdir(output_community_folder)

    node_size = data_process_lib.calculate_node_size_for_drawing(G)
    layout = data_process_lib.try_calculate_layout_with_cache(G)

    # community
    node_color_dict = {}
    node_labels = {}
    comp = nx.community.girvan_newman(G, most_valuable_edge=most_valuable_edge)
    limited = itertools.takewhile(lambda c: len(c) <= number_of_community, comp)
    for communities in limited:
        final_communities = tuple(sorted(c) for c in communities)
        k = len(final_communities)
        for i, single_community in enumerate(final_communities):
            print(f"community_{k} {i} -- {single_community}")
        for i, community in enumerate(final_communities):
            for single_node in community:
                node_color_dict[int(single_node)] = i
                node_labels[single_node] = f"{single_node}"

        # save final communities
        data_process_lib.save_data(final_communities, os.path.join(output_community_folder, f"community_{k}.pkl"))

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

        fig.savefig(os.path.join(output_community_folder, f"community_{k}.pdf"), pad_inches=0)
        plt.close(fig)

