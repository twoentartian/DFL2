import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import colorsys
import os
import re
import cv2
import multiprocessing
import data_process_lib
import argparse

config_file_path = 'simulator_config.json'
accuracy_file_path = 'accuracy.csv'
peer_change_file_path = 'peer_change_record.txt'
draw_interval = 1
fps = 4
dpi = 200
override_existing_cache = True
HSV_H_start = 40
HSV_H_end = 256

video_cache_path = "./video_cache"


def save_fig(G: nx.Graph, tick, save_name, node_accuracies, layout, node_labels, node_size, with_labels, override_existing=False):
    if not override_existing and os.path.exists(save_name):
        return

    fig = plt.figure(frameon=False)
    fig.set_size_inches(12, 12)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.text(0, 0, "tick = " + str(tick))
    cmap = matplotlib.colormaps.get_cmap('viridis')
    normalize = matplotlib.colors.Normalize(vmin=0, vmax=1)
    node_colors = [cmap(normalize(node_accuracy)) for node_accuracy in node_accuracies]

    nx.draw(G, node_color=node_colors, with_labels=with_labels, pos=layout, font_color='k', labels=node_labels, alpha=0.7, linewidths=0.1, width=0.1, font_size=8, node_size=node_size)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=normalize)
    sm.set_array([0, 1])
    fig.colorbar(sm, ax=ax, orientation='vertical', label='Accuracy Values', shrink=0.4)
    fig.savefig(save_name, dpi=dpi, pad_inches=0)
    plt.close(fig)


if __name__ == "__main__":
    # parser args
    parser = argparse.ArgumentParser(description="generate a video for accuracy trends, put this script with \"simulator_config.json\"", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--override_cache", help="override images cache?", type=bool)
    args = parser.parse_args()
    config = vars(args)
    override_cache = False
    if config["override_cache"]:
        override_cache = True

    G = data_process_lib.load_graph_from_simulation_config(config_file_path)

    accuracy_df = data_process_lib.load_csv_with_parquet_acceleration(accuracy_file_path, False)

    peer_change_file_exists = os.path.exists(peer_change_file_path)
    peer_change_list = []
    if peer_change_file_exists:
        peer_change_file = open(peer_change_file_path, "r+")
        peer_change_content = peer_change_file.readlines()
        for line in peer_change_content:
            operation = 0
            result = re.findall('tick:\d+', line)
            tick = int(result[0][5:])
            result = re.findall('\s\w*\(accuracy', line)
            lhs_node = result[0][1:-9]
            result = re.findall('\s\w*\(buffer', line)
            rhs_node = result[0][1:-7]
            add_ = re.findall('\sadd\s', line)
            delete_ = re.findall('\sdelete\s', line)
            if add_ and not delete_:
                operation = 1
            if not add_ and delete_:
                operation = 2
            peer_change_list.append({'tick': tick, 'lhs_node': lhs_node, 'rhs_node': rhs_node, 'operation': operation})
        peer_change_file.close()

    total_tick = len(accuracy_df.index)
    draw_counter = 0
    tick_to_draw = []
    for tick in accuracy_df.index:
        if draw_counter >= draw_interval-1:
            draw_counter = 0
            tick_to_draw.append(tick)
        else:
            draw_counter = draw_counter + 1
            continue

    # layout = nx.spring_layout(G, k=5/math.sqrt(G.order()))
    # layout = nx.circular_layout(G)
    # layout = nx.spectral_layout(G)
    # layout = nx.kamada_kawai_layout(G)
    # layout = nx.shell_layout(G)
    # layout = nx.random_layout(G)
    layout = nx.nx_agraph.graphviz_layout(G)

    node_name = G.nodes
    peer_change_list_index = 0
    if not os.path.isdir(video_cache_path):
        os.mkdir(video_cache_path)

    pool = multiprocessing.Pool(processes=os.cpu_count())

    N = len(G.nodes)
    print(f"N={N}")
    node_size = int(50000/N)
    node_size = max(10, node_size)
    print(f"draw_node_size={node_size}")
    with_labels = True
    if N > 300:
        with_labels = False
    print(f"draw_with_labels={with_labels}")

    for tick in tick_to_draw:
        print("processing tick: " + str(tick))
        if os.path.exists(os.path.join(video_cache_path, str(tick) + ".png")) and not override_existing_cache:
            continue

        node_labels = {}

        # node edge
        while peer_change_list_index < len(peer_change_list) and tick > peer_change_list[peer_change_list_index]['tick']:
            current_peer_change = peer_change_list[peer_change_list_index]
            if current_peer_change['operation'] == 1:  # add edge
                G.add_edge(current_peer_change['lhs_node'], current_peer_change['rhs_node'])
            if current_peer_change['operation'] == 2:  # delete edge
                G.remove_edge(current_peer_change['lhs_node'], current_peer_change['rhs_node'])
            peer_change_list_index = peer_change_list_index + 1

        # node color
        node_accuracies = []
        for node in G.nodes:
            accuracy = accuracy_df.loc[tick, node]
            node_accuracies.append(accuracy)
            node_labels[node] = str(accuracy)

        # save to files
        pool.apply_async(save_fig, (G.copy(), tick, os.path.join(video_cache_path, str(tick) + ".png"), node_accuracies, layout, node_labels, node_size, with_labels, override_cache))

        # save the map
        if tick == tick_to_draw[0]:
            maximum_degree_node, maximum_degree = max(G.degree)
            node_color = []
            node_labels = {}
            for node in G.nodes:
                (r, g, b) = colorsys.hsv_to_rgb(HSV_H_start / 256 + (1 - HSV_H_start / 256) * G.degree[node] / maximum_degree, 0.5, 1.0)
                node_color.append([r, g, b])
                node_labels[node] = str(node) + "(" + str(G.degree[node]) + ")"
            pool.apply_async(save_fig, (G.copy(), tick, "map.pdf", node_accuracies, layout, node_labels, node_size, with_labels, override_cache))

    pool.close()
    pool.join()

    # let opencv generate video
    print("generating video")

    first_img = cv2.imread(os.path.join(video_cache_path, str(tick_to_draw[0]) + ".png"))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width, channel = first_img.shape
    video = cv2.VideoWriter('video.mp4', fourcc, fps, (width, height))
    for tick in tick_to_draw:
        img = cv2.imread(os.path.join(video_cache_path, str(tick) + ".png"))
        video.write(img)
    video.release()

