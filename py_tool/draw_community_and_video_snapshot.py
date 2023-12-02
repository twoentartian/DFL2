import os
import sys

import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import data_process_lib

save_name = "community_and_video_snapshot"

config_file_path = 'simulator_config.json'
accuracy_file_path = 'accuracy.csv'

data_list = {}
data_list["community"] = {
    "type": "community_map",
    "community_file": "final_communities.pkl",
    "plot_loc": [0, 0],
    "title": "communities identified by Girvan-Newman algorithm",
}

data_list["tick_0"] = {
    "type": "snapshot",
    "snapshot_tick": 1200,
    "plot_loc": [0, 1],
    "title": f"vertex learning status at tick 1200",
}

data_list["tick_1"] = {
    "type": "snapshot",
    "snapshot_tick": 2140,
    "plot_loc": [0, 2],
    "title": "vertex learning status at tick 2140",
}

data_list["tick_2"] = {
    "type": "snapshot",
    "snapshot_tick": 3360,
    "plot_loc": [0, 3],
    "title": "vertex learning status at tick 3360",
}



plot_row = 1
plot_col = 4
figsize = 6
figsize_col = figsize * plot_col
figsize_row = figsize * plot_row

if __name__ == "__main__":
    # load accuracy file
    accuracy_df = data_process_lib.load_csv_with_parquet_acceleration(accuracy_file_path, False)
    G = data_process_lib.load_graph_from_simulation_config(config_file_path)

    layout = nx.nx_agraph.graphviz_layout(G)

    N = len(G.nodes)
    ratio = (figsize*figsize/100)
    node_size = int((50000/N) * ratio)
    node_size = max(10*ratio, node_size)

    # draw
    whole_fig, whole_axs = plt.subplots(plot_row, plot_col, figsize=(figsize_col, figsize_row), squeeze=False)
    for [data_name, data] in data_list.items():
        axs:plt.Axes = whole_axs[tuple(data["plot_loc"])]
        axs.set_title(data["title"])
        if data["type"] == "community_map":
            # load community file
            community_file_path = data["community_file"]
            communities = data_process_lib.try_load_data(community_file_path)
            if communities is None:
                print(f"{community_file_path} is invalid")
                exit(-1)
            node_labels = {}
            for i, community in enumerate(communities):
                for single_node in community:
                    node_labels[single_node] = data_process_lib.int_to_text(i+1)

            # get color from community file
            node_color_dict = {}
            for i, community in enumerate(communities):
                for single_node in community:
                    node_color_dict[int(single_node)] = i
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

            # draw
            axs.set_axis_off()
            nx.draw(G, node_color=node_color_draw, ax=axs, with_labels=True, pos=layout,
                    font_color='k', labels=node_labels, alpha=0.7, linewidths=0.1, width=0.1,
                    font_size=8, node_size=node_size)

        if data["type"] == "snapshot":
            tick = data["snapshot_tick"]
            axs.text(0, 0, f"tick={tick}")
            node_accuracies = []
            for node in G.nodes:
                accuracy = accuracy_df.loc[tick, node]
                node_accuracies.append(accuracy)
            cmap = matplotlib.colormaps.get_cmap('viridis')
            normalize = matplotlib.colors.Normalize(vmin=0, vmax=1)
            node_colors = [cmap(normalize(node_accuracy)) for node_accuracy in node_accuracies]

            nx.draw(G, node_color=node_colors, ax=axs, with_labels=False, pos=layout,
                    font_color='k', labels=None, alpha=0.7, linewidths=0.1, width=0.1,
                    font_size=8, node_size=node_size)

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=normalize)
            sm.set_array([0, 1])
            whole_fig.colorbar(sm, ax=axs, location="left",
                               label='Learning status', shrink=0.4, pad=0)

    whole_fig.tight_layout(pad=-5)
    whole_fig.savefig(f'./{save_name}.pdf', pad_inches=0, bbox_inches='tight')
    plt.close(whole_fig)

