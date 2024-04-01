import os
import sys

import numpy as np
import pandas
import matplotlib.pyplot as plt
import json
import networkx as nx
from scipy import stats

import draw_info

methods_to_test = [ {"func":nx.closeness_centrality, "name":"closeness_centrality"},
                    {"func":nx.current_flow_closeness_centrality, "name":"current_flow_closeness_centrality"},
                    {"func":nx.betweenness_centrality, "name":"betweenness_centrality"},
                    {"func":nx.current_flow_betweenness_centrality, "name":"current_flow_betweenness_centrality"},
                    # {"func":nx.communicability_betweenness_centrality, "name":"communicability_betweenness_centrality"},
                    # {"func":nx.harmonic_centrality, "name":"harmonic_centrality"},
                    # {"func":nx.global_reaching_centrality, "name":"global_reaching_centrality"},
                    # {"func":nx.second_order_centrality, "name":"second_order_centrality"},
                    # {"func":nx.laplacian_centrality, "name":"laplacian_centrality"},
                    ]
use_ratio = False   #False - use freeman's graph centrality measure.
                    #True - use Zaid's hypothesis
if use_ratio:
    save_path = "herd_effect_and_graph_property_ratio.csv"
else:
    save_path = "herd_effect_and_graph_property_subtraction.csv"


def calculate_herd_effect_delay(arg_accuracy_df: pandas.DataFrame, arg_model_weight_diff_df: pandas.DataFrame):
    average_accuracy: pandas.Series = arg_accuracy_df.mean(axis=1)
    average_accuracy_diff = average_accuracy.diff()
    average_accuracy_diff.dropna(inplace=True)
    largest_diff = average_accuracy_diff.nlargest(10)
    largest_indexes = largest_diff.index
    for i in largest_indexes:
        if i > draw_info.first_average_time*2:
            return i


def sum_of_deviations_from_max(values):
    max_val = max(values)
    if use_ratio:
        return sum(v / max_val for v in values)
    else:
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


def calculate_slope_intercept_for_loglog(x, y):
    from sklearn.linear_model import LinearRegression
    x = np.log(x)
    y = np.log(y)
    x = x.values
    y = y.values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return slope, intercept, r_value, p_value, std_err


if __name__ == "__main__":
    assert len(draw_info.folders) <= draw_info.row * draw_info.col
    folder_names_set = set()
    for folder_index in range(len(draw_info.folders)):
        folder = draw_info.folders[folder_index]
        assert not (folder in folder_names_set)
        folder_names_set.add(folder)

    if os.path.exists(save_path):
        print(f"{save_path} already exists, loading it")
        output_df = pandas.read_csv(save_path)
    else:
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


                # # eigenvector_centrality = graph_centrality(G, nx.eigenvector_centrality)    #exception
                # # katz_centrality = graph_centrality(G, nx.katz_centrality)    #exception
                # closeness_centrality = graph_centrality(G, nx.closeness_centrality)
                # current_flow_closeness_centrality = graph_centrality(G, nx.current_flow_closeness_centrality)
                # betweenness_centrality = graph_centrality(G, nx.betweenness_centrality)
                # current_flow_betweenness_centrality = graph_centrality(G, nx.current_flow_betweenness_centrality)
                # communicability_betweenness_centrality = graph_centrality(G, nx.communicability_betweenness_centrality)
                # harmonic_centrality = graph_centrality(G, nx.harmonic_centrality)
                # # dispersion = graph_centrality(G, nx.dispersion)    #exception
                # global_reaching_centrality = graph_centrality(G, nx.global_reaching_centrality)
                # # percolation_centrality = graph_centrality(G, nx.percolation_centrality)    #exception
                # second_order_centrality = graph_centrality(G, nx.second_order_centrality)
                # # trophic_levels = graph_centrality(G, nx.trophic_levels)    #exception
                # # trophic_incoherence_parameter = graph_centrality(G, nx.trophic_incoherence_parameter)    #exception
                # # voterank = graph_centrality(G, nx.voterank)
                # laplacian_centrality = graph_centrality(G, nx.laplacian_centrality)

                row_data = {}
                row_data["folder"] = folder
                row_data["each_test_result_folder"] = each_test_result_folder
                row_data["herd_effect_delay"] = herd_effect_delay
                for method in methods_to_test:
                    func = method["func"]
                    method_name = method["name"]
                    row_data[method_name] = graph_centrality(G, func)
                final_output_df_content.append(row_data)

        output_df = pandas.DataFrame(final_output_df_content)
        output_df.to_csv(save_path)

    for method in methods_to_test:
        func = method["func"]
        method_name = method["name"]
        fig, axs = plt.subplots(1, 1, figsize=(10, 10), squeeze=False)
        axs[0, 0].scatter(output_df[method_name], output_df['herd_effect_delay'], label=f'herd_effect_delay vs {method_name}')
        # axs[0, 0].legend()
        axs[0, 0].grid()
        axs[0, 0].set_yscale('log')
        axs[0, 0].set_xscale('log')

        # if method_name == "current_flow_betweenness_centrality":
        if True:
            slope, intercept, r_value, p_value, std_err = calculate_slope_intercept_for_loglog(output_df[method_name], output_df['herd_effect_delay'])
            print(f"{method_name} -- slope: {slope}, intercept: {intercept}, stderr: {std_err}")
            asymptote_x = np.log(output_df[method_name])
            asymptote_y = [i * slope + intercept for i in asymptote_x]
            asymptote_x = np.exp(asymptote_x)
            asymptote_y = np.exp(asymptote_y)
            line, = axs[0, 0].plot(asymptote_x, asymptote_y, alpha=0.8, c="tab:orange", label=f'asymptote: slope={slope:.2f}')
            line.set_dashes((2,1))

        for x,y,test_name in zip(output_df[method_name], output_df['herd_effect_delay'], output_df.index):
            axs[0, 0].annotate(f"{test_name}", (x,y),textcoords="offset points",xytext=(3,3),ha='center')
        if use_ratio:
            fig.savefig(f"ratio_graph_centrality_{method_name}.pdf")
        else:
            fig.savefig(f"subtraction_graph_centrality_{method_name}.pdf")

