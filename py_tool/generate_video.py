import json
import networkx as nx
import numpy as np
import pandas
import numpy
import matplotlib.pyplot as plt
import tkinter
import colorsys
import os
import re
import cv2
import multiprocessing
import math

config_file_path = 'simulator_config.json'
accuracy_file_path = 'accuracy.csv'
peer_change_file_path = 'peer_change_record.txt'
draw_per_row = 1
fps = 2
dpi = 200
override_existing_cache = True
HSV_H_start = 40
HSV_H_end = 256

video_cache_path = "./video_cache"


def save_fig(G, tick, save_name, node_color, layout, node_labels):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(12, 12)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.text(0, 0, "tick = " + str(tick))
    nx.draw(G, node_color=node_color, with_labels=True, pos=layout, font_color='k', labels=node_labels, alpha=1.0, linewidths=0.1, width=0.5, font_size=8, node_size=300)
    fig.savefig(os.path.join(save_name), dpi=dpi, pad_inches=0)
    plt.close(fig)

    # b = figure.get_window_extent()
    # img = numpy.array(figure.canvas.buffer_rgba())
    # img = img[int(b.y0):int(b.y1), int(b.x0):int(b.x1), :]
    # img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)


if __name__ == "__main__":
    config_file = open(config_file_path)
    config_file_content = config_file.read()
    config_file_json = json.loads(config_file_content)
    topology = config_file_json['node_topology']
    peer_control_enabled = config_file_json['services']['peer_control_service']['enable']
    nodes = config_file_json['nodes']

    accuracy_df = pandas.read_csv(accuracy_file_path, index_col=0, header=0)

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
        if draw_counter >= draw_per_row:
            draw_counter = 0
            tick_to_draw.append(tick)
        else:
            draw_counter = draw_counter + 1
            continue

    peer_count = {}
    G = nx.Graph()
    for single_node in nodes:
        G.add_node(single_node['name'])
        peer_count[single_node['name']] = 0

    if not peer_control_enabled:
        for singleItem in topology:
            unDirLink = singleItem.split('--')
            if len(unDirLink) != 1:
                G.add_edge(unDirLink[0], unDirLink[1])
                peer_count[unDirLink[0]] = peer_count[unDirLink[0]] + 1
                peer_count[unDirLink[1]] = peer_count[unDirLink[1]] + 1

            dirLink = singleItem.split('->')
            if len(dirLink) != 1:
                G.add_edge(dirLink[0], dirLink[1])
                peer_count[unDirLink[0]] = peer_count[unDirLink[0]] + 1

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

    for tick in tick_to_draw:
        print("processing tick: " + str(tick))
        if os.path.exists(os.path.join(video_cache_path, str(tick) + ".png")) and not override_existing_cache:
            continue

        node_color = []
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
        for node in G.nodes:
            accuracy = accuracy_df.loc[tick, node]
            # (r, g, b) = colorsys.hsv_to_rgb(HSV_H_start / 256 + (1 - HSV_H_start / 256) * accuracy, 0.5, 1.0)
            (r, g, b) = colorsys.hsv_to_rgb(accuracy, 0.5, 1.0)
            node_color.append([r, g, b])
            node_labels[node] = str(accuracy)

        # save to files
        pool.apply_async(save_fig, (G.copy(), tick, os.path.join(video_cache_path, str(tick) + ".png"), node_color, layout, node_labels))

        # save the map
        if tick == tick_to_draw[0]:
            maximum_peer = max(peer_count.values())
            node_color = []
            node_labels = {}
            for node in G.nodes:
                (r, g, b) = colorsys.hsv_to_rgb(HSV_H_start / 256 + (1 - HSV_H_start / 256) * peer_count[node] / maximum_peer, 0.5, 1.0)
                node_color.append([r, g, b])
                node_labels[node] = str(node) + "(" + str(peer_count[node]) + ")"
            pool.apply_async(save_fig, (G.copy(), tick, "map.pdf", node_color, layout, node_labels))

    pool.close()
    pool.join()

    # generate the map

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
