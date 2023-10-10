import os
import sys

import numpy
import pandas
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

import draw_info
import json
import networkx as nx

first_average_time = 20


def calculate_herd_effect_delay(arg_accuracy_df: pandas.DataFrame, arg_model_weight_diff_df: pandas.DataFrame):
    herd_effect_delay_tick = 0

    average_accuracy: pandas.Series = arg_accuracy_df.mean(axis=1)
    average_accuracy_diff = average_accuracy.diff()
    average_accuracy_diff.dropna(inplace=True)
    # herd_effect_delay_tick = average_accuracy_diff.idxmax()
    largest_diff = average_accuracy_diff.nlargest(10)
    largest_indexes = largest_diff.index
    for i in largest_indexes:
        if i > first_average_time*2:
            return i


def moving_average(data, window_size):
    return data.rolling(window=window_size, min_periods=1).mean()


if __name__ == "__main__":
    assert len(draw_info.folders) <= draw_info.row * draw_info.col
    folder_names_set = set()
    for folder_index in range(len(draw_info.folders)):
        folder = draw_info.folders[folder_index]
        assert not (folder in folder_names_set)
        folder_names_set.add(folder)

    output = []
    for folder_index in range(len(draw_info.folders)):
        folder = draw_info.folders[folder_index]
        subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]
        assert len(subfolders) != 0
        for each_test_result_folder in subfolders:

            accuracy_file_path = each_test_result_folder + '/accuracy.csv'
            accuracy_df = pandas.read_csv(accuracy_file_path, index_col=0, header=0)

            weight_diff_file_path = each_test_result_folder + '/model_weight_diff.csv'
            weight_diff_df = pandas.read_csv(weight_diff_file_path, index_col=0, header=0)

            herd_effect_delay = calculate_herd_effect_delay(accuracy_df, weight_diff_df)

            config_file = open(each_test_result_folder + '/simulator_config.json')
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
            max_k = sorted(G.degree, key=lambda x: x[1], reverse=True)[0][1]
            N = len(G.nodes)
            max_k_normalized = max_k / N
            constant = herd_effect_delay / max_k_normalized**-1.775
            print(f"constant = {constant} = {max_k_normalized} * {herd_effect_delay}")
            output.append({"name":each_test_result_folder, "herd_delay":herd_effect_delay, "max_k":max_k, "graph_size":N, "max_k_normalized":max_k_normalized, "constant": constant})
        output_df = pandas.DataFrame(output)
        output_df.to_csv("max_k_herd_delay.csv")

        output_df['smoothed_constant'] = moving_average(output_df['constant'], 15)

        whole_fig, whole_axs = plt.subplots(1, 1, figsize=(10, 10), squeeze=False)
        whole_axs[0, 0].plot(output_df["name"], output_df["constant"])
        whole_axs[0, 0].plot(output_df["name"], output_df["smoothed_constant"])
        whole_fig.savefig('max_k_herd_delay_constant.pdf')
        plt.close(whole_fig)