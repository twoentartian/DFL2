import os
import sys

import matplotlib.pyplot as plt

import draw_info

simulator_config_file_name = "simulator_config.json"
save_name = "degree_distribution"


def draw_single_degree_distribution(config_path, axis: plt.Axes):
    import json
    config_file = open(config_path)
    config_file_content = config_file.read()
    config_file_json = json.loads(config_file_content)
    topology = config_file_json['node_topology']

    peer_count_of_each_node = {}
    for singleItem in topology:
        un_dir_link = singleItem.split('--')
        if len(un_dir_link) == 2:
            lhs_node = int(un_dir_link[0])
            rhs_node = int(un_dir_link[1])
            if lhs_node not in peer_count_of_each_node.keys():
                peer_count_of_each_node[lhs_node] = set()
            if rhs_node not in peer_count_of_each_node.keys():
                peer_count_of_each_node[rhs_node] = set()
            peer_count_of_each_node[lhs_node].add(rhs_node)
            peer_count_of_each_node[rhs_node].add(lhs_node)
        dir_link = singleItem.split('->')
        if len(dir_link) == 2:
            lhs_node = int(dir_link[0])
            rhs_node = int(dir_link[1])
            if lhs_node not in peer_count_of_each_node.keys():
                peer_count_of_each_node[lhs_node] = set()
            peer_count_of_each_node[lhs_node].add(rhs_node)

    degree_count = {}
    for (node_name, peers) in peer_count_of_each_node.items():
        peer_count = len(peers)
        if peer_count not in degree_count.keys():
            degree_count[peer_count] = 0
        degree_count[peer_count] += 1

    node_plot = list(degree_count.keys())
    degree_plot = list(degree_count.values())
    axis.bar(node_plot, degree_plot)
    axis.grid()
    axis.set_title('Subplot ' + str(folder_index + 1) + " " + draw_info.titles[folder_index])
    axis.set_xlabel('degree')
    axis.set_ylabel('count')
    axis.set_yscale('log')


if __name__ == "__main__":
    assert len(draw_info.folders) <= draw_info.row * draw_info.col
    fig, whole_axis = plt.subplots(draw_info.row, draw_info.col, squeeze=False, figsize=(draw_info.col*8, draw_info.row*8))
    for folder_index in range(len(draw_info.folders)):
        current_col = folder_index % draw_info.col
        current_row = folder_index // draw_info.col
        folder = draw_info.folders[folder_index]
        sub_folders = [f.path for f in os.scandir(folder) if f.is_dir()]
        assert len(sub_folders) != 0
        first_sub_folder_path = sub_folders[0]
        simulation_config_file = os.path.join(first_sub_folder_path, simulator_config_file_name)
        current_axis = whole_axis[current_row, current_col]
        draw_single_degree_distribution(simulation_config_file, current_axis)
    fig.savefig(save_name + ".pdf")
    fig.savefig(save_name + ".jpg", dpi=800)
