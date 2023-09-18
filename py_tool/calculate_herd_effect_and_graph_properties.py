import os
import sys

import numpy
import pandas
import matplotlib.pyplot as plt
import json
import networkx as nx

import draw_info

first_average_time = 20


def calculate_herd_effect_delay(arg_accuracy_df: pandas.DataFrame, arg_model_weight_diff_df: pandas.DataFrame):
    average_accuracy: pandas.Series = arg_accuracy_df.mean(axis=1)
    average_accuracy_diff = average_accuracy.diff()
    average_accuracy_diff.dropna(inplace=True)
    largest_diff = average_accuracy_diff.nlargest(10)
    largest_indexes = largest_diff.index
    for i in largest_indexes:
        if i > first_average_time*2:
            return i


def sum_of_deviations_from_max(values):
    max_val = max(values)
    return sum(max_val - v for v in values)


def graph_centrality(G, vertex_centrality_func, normalized = False):
    vertex_centrality = vertex_centrality_func(G)
    if vertex_centrality_func == nx.global_reaching_centrality:
        return vertex_centrality

    output = sum_of_deviations_from_max(list(vertex_centrality.values()))
    if normalized:
        star_graph = nx.star_graph(G.number_of_nodes())
        star_centrality = vertex_centrality_func(star_graph)
        output = output / sum_of_deviations_from_max(list(star_centrality.values()))
    return output


if __name__ == "__main__":
    assert len(draw_info.folders) <= draw_info.row * draw_info.col
    folder_names_set = set()
    for folder_index in range(len(draw_info.folders)):
        folder = draw_info.folders[folder_index]
        assert not (folder in folder_names_set)
        folder_names_set.add(folder)

    final_output_df_content = []
    for folder_index in range(len(draw_info.folders)):
        folder = draw_info.folders[folder_index]
        subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]
        assert len(subfolders) != 0
        for each_test_result_folder in subfolders:
            print(f'processing {each_test_result_folder} in {folder}')
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
            # eigenvector_centrality = graph_centrality(G, nx.eigenvector_centrality)    #exception
            # katz_centrality = graph_centrality(G, nx.katz_centrality)    #exception
            closeness_centrality = graph_centrality(G, nx.closeness_centrality)
            current_flow_closeness_centrality = graph_centrality(G, nx.current_flow_closeness_centrality)
            betweenness_centrality = graph_centrality(G, nx.betweenness_centrality)
            current_flow_betweenness_centrality = graph_centrality(G, nx.current_flow_betweenness_centrality)
            communicability_betweenness_centrality = graph_centrality(G, nx.communicability_betweenness_centrality)
            harmonic_centrality = graph_centrality(G, nx.harmonic_centrality)
            # dispersion = graph_centrality(G, nx.dispersion)    #exception
            global_reaching_centrality = graph_centrality(G, nx.global_reaching_centrality)
            # percolation_centrality = graph_centrality(G, nx.percolation_centrality)    #exception
            second_order_centrality = graph_centrality(G, nx.second_order_centrality)
            # trophic_levels = graph_centrality(G, nx.trophic_levels)    #exception
            # trophic_incoherence_parameter = graph_centrality(G, nx.trophic_incoherence_parameter)    #exception
            # voterank = graph_centrality(G, nx.voterank)
            laplacian_centrality = graph_centrality(G, nx.laplacian_centrality)

            final_output_df_content.append([folder, each_test_result_folder, herd_effect_delay, closeness_centrality, current_flow_closeness_centrality,
                                            betweenness_centrality, current_flow_betweenness_centrality, communicability_betweenness_centrality,
                                            harmonic_centrality, global_reaching_centrality, second_order_centrality, laplacian_centrality])
    output_df = pandas.DataFrame(final_output_df_content, columns=["folder", "each_test_result_folder", "herd_effect_delay", "closeness_centrality", "current_flow_closeness_centrality",
                                                                   "betweenness_centrality", "current_flow_betweenness_centrality", "communicability_betweenness_centrality",
                                                                   "harmonic_centrality", "global_reaching_centrality", "second_order_centrality", "laplacian_centrality"])
    output_df.to_csv("herd_effect_and_graph_property.csv")






