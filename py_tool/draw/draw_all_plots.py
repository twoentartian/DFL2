import os
import sys

import numpy
import pandas
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

import draw_info
import data_process_lib
import json
import networkx as nx

reduce_image_file_size = True
lines_limit_per_image = 50


save_name = "draw"
draw_model_weight_diff = True
draw_topology_map = True
draw_topology_map_in_bitmap_format = False

herd_effect_weight_diff_reference_layer = "conv2"
herd_effect_draw_with_size = False

maximum_tick = draw_info.maximum_tick


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")


def calculate_herd_effect_delay(arg_accuracy_df: pandas.DataFrame, arg_model_weight_diff_df: pandas.DataFrame):
    herd_effect_delay_tick = 0

    # # try the weight diff method
    # diff_series: pandas.Series = arg_model_weight_diff_df[herd_effect_weight_diff_reference_layer]
    # window_size = 1
    # while True:
    #     diff_series_averaged = diff_series.rolling(window=window_size).mean()
    #     diff_series_averaged.dropna(inplace=True)
    #     peaks, _ = find_peaks(diff_series_averaged)
    #     if len(peaks) == 1:
    #         # fig1, ax1 = plt.subplots()
    #         # ax1.plot(diff_series_averaged)
    #         # ax1.set_yscale('log')
    #         # fig1.show()
    #
    #         herd_effect_delay_tick = diff_series_averaged.iloc[[peaks[0] - window_size // 2 - 1]].index[0]   # shift back to the mid of the mean averaging window
    #         break
    #     if len(peaks) == 0:
    #         # we failed, we should try using the accuracy graph
    #         break
    #     window_size = window_size + 2
    # if herd_effect_delay_tick != 0:
    #     return herd_effect_delay_tick

    # # try the min accuracy method
    # min_accuracy: pandas.Series = arg_accuracy_df.min(axis=1)
    # min_accuracy_averaged = min_accuracy.rolling(window=7).mean()
    # herd_effect_delay_tick = min_accuracy_averaged[min_accuracy_averaged > 0.1].first_valid_index()

    # try the maximum of derivative
    average_accuracy: pandas.Series = arg_accuracy_df.mean(axis=1)
    average_accuracy_diff = average_accuracy.diff()
    average_accuracy_diff.dropna(inplace=True)
    # herd_effect_delay_tick = average_accuracy_diff.idxmax()
    largest_diff = average_accuracy_diff.nlargest(10)
    largest_indexes = largest_diff.index
    for i in largest_indexes:
        if i > draw_info.first_average_time*2:
            return i


if __name__ == "__main__":
    assert len(draw_info.folders) <= draw_info.row * draw_info.col
    folder_names_set = set()
    for folder_index in range(len(draw_info.folders)):
        folder = draw_info.folders[folder_index]
        assert not (folder in folder_names_set)
        folder_names_set.add(folder)

    flag_generate_whole = query_yes_no('do you want to generate the whole figure?')
    if flag_generate_whole:
        figsize_col = 5 * draw_info.col
        figsize_row = 2.5 * draw_info.row
        plot_row = draw_info.row
        plot_col = draw_info.col
        number_of_plot_per_row = 1
        if draw_model_weight_diff:
            figsize_row = figsize_row + figsize_row
            plot_row = plot_row + draw_info.row
            number_of_plot_per_row = number_of_plot_per_row + 1
        if draw_topology_map:
            figsize_row = figsize_row + figsize_row
            plot_row = plot_row + draw_info.row
            number_of_plot_per_row = number_of_plot_per_row + 1

        whole_fig, whole_axs = plt.subplots(plot_row, plot_col, figsize=(figsize_col, figsize_row), squeeze=False)
        topology_graphs = []
        herd_effect_delay_df = pandas.DataFrame(columns=['network_name',
                                                         'herd_effect_delay',
                                                         'size',
                                                         'max_degree',
                                                         "folder_name",
                                                         "size_of_giant_island",
                                                         "CurrentFlowBetweenCentrality_GiantIsland"])
        for folder_index in range(len(draw_info.folders)):
            current_col = folder_index % draw_info.col
            current_row = folder_index // draw_info.col

            folder = draw_info.folders[folder_index]

            subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]
            assert len(subfolders) != 0
            first_accuracy_df = pandas.DataFrame()
            first_weight_diff_df = pandas.DataFrame()
            is_first_dataframe = True
            first_to_draw_topology = True

            herd_effect_delay_results = []
            max_degree = []
            size_of_largest_island = []
            current_flow_betweenness_centrality = []
            first_herd_effect_delay = 0
            for each_test_result_folder in subfolders:
                # load the topology
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

                # degree
                degrees = dict(G.degree())
                max_degree.append(max(degrees.values()))
                # largest island
                giant_component = G.subgraph(max(nx.connected_components(G), key=len))
                size_of_largest_island.append(len(giant_component.nodes))
                # current flow betweenness centrality of the giant component
                centrality = data_process_lib.graph_centrality_normalized(giant_component, nx.current_flow_betweenness_centrality)
                current_flow_betweenness_centrality.append(centrality)

                accuracy_file_path = each_test_result_folder + '/accuracy.csv'
                accuracy_df = pandas.read_csv(accuracy_file_path, index_col=0, header=0)
                # print(accuracy_df)

                weight_diff_file_path = each_test_result_folder + '/model_weight_diff.csv'
                weight_diff_df = pandas.read_csv(weight_diff_file_path, index_col=0, header=0)
                # print(weight_diff_df)

                herd_effect_delay = calculate_herd_effect_delay(accuracy_df, weight_diff_df)
                herd_effect_delay_results.append(herd_effect_delay)

                new_row = pandas.DataFrame({'herd_effect_delay': herd_effect_delay,
                                            "network_name": draw_info.titles[folder_index],
                                            "folder_name": each_test_result_folder,
                                            "size": len(accuracy_df.columns),
                                            "max_degree": max(degrees.values()),
                                            "size_of_giant_island": len(giant_component.nodes),
                                            "CurrentFlowBetweenCentrality_GiantIsland": centrality,
                },index=[0])
                herd_effect_delay_df = pandas.concat([herd_effect_delay_df.loc[:], new_row]).reset_index(drop=True)

                if first_to_draw_topology:
                    first_to_draw_topology = False
                    topology_graphs.append(G)
                    first_accuracy_df = accuracy_df
                    first_weight_diff_df = weight_diff_df
                    first_herd_effect_delay = herd_effect_delay

            number_of_nodes = len(first_accuracy_df.columns)
            print(str(folder) + " --- herd effect delay = " + str(herd_effect_delay_results))
            average_herd_delay = sum(herd_effect_delay_results)/len(herd_effect_delay_results)
            if len(subfolders) == 1:
                max_degree = max_degree[0]
                size_of_largest_island = size_of_largest_island[0]
                current_flow_betweenness_centrality = current_flow_betweenness_centrality[0]

            accuracy_x = first_accuracy_df.index
            accuracy_df_len = len(first_accuracy_df)
            weight_diff_x = first_weight_diff_df.index
            weight_diff_df_len = len(first_weight_diff_df)

            accuracy_axis = whole_axs[current_row * number_of_plot_per_row, current_col]

            if maximum_tick >= accuracy_x[len(accuracy_x) - 1]:
                end_accuracy_x = len(accuracy_x) - 1
            else:
                end_accuracy_x = next(
                    k for k, value in enumerate(accuracy_x) if value > maximum_tick)  # find the end of axis

            if not reduce_image_file_size:
                # normal plot
                for _col in first_accuracy_df.columns:
                    accuracy_axis.plot(accuracy_x[0:end_accuracy_x], first_accuracy_df[_col].iloc[0:end_accuracy_x], label=_col, alpha=0.75)
            else:
                if lines_limit_per_image < len(first_accuracy_df.columns):
                    ratio = 1 / len(first_accuracy_df.columns)
                    target_ratio_per_draw = 1 / lines_limit_per_image
                    current_ratio = 0
                    for _col in first_accuracy_df.columns:
                        if current_ratio > target_ratio_per_draw:
                            accuracy_axis.plot(accuracy_x[0:end_accuracy_x], first_accuracy_df[_col].iloc[0:end_accuracy_x], label=_col, alpha=0.75)
                            current_ratio -= target_ratio_per_draw
                        current_ratio += ratio

                else:
                    for _col in first_accuracy_df.columns:
                        accuracy_axis.plot(accuracy_x[0:end_accuracy_x], first_accuracy_df[_col].iloc[0:end_accuracy_x], label=_col, alpha=0.75)

            accuracy_axis.axvline(x=first_herd_effect_delay, color='r', label='herd effect delay')

            accuracy_axis.grid()
            accuracy_axis.legend(ncol=5)
            if len(first_accuracy_df.columns) > 10:
                accuracy_axis.legend().remove()
            accuracy_axis.set_title('Subplot ' + str(folder_index + 1) + 'a - accuracy: ' + draw_info.titles[folder_index])
            accuracy_axis.set_xlabel('time (tick)')
            accuracy_axis.set_ylabel('accuracy (0-1)')
            # accuracy_axis.set_xlim([0, final_accuracy_df.index[end_accuracy_x]])
            accuracy_axis.set_xlim([0, maximum_tick])
            accuracy_axis.set_ylim([0, 1])

            if draw_model_weight_diff:
                weight_diff_axis = whole_axs[current_row * number_of_plot_per_row + 1, current_col]
                if maximum_tick >= weight_diff_x[len(weight_diff_x) - 1]:
                    end_weight_diff_x = len(weight_diff_x) - 1
                else:
                    end_weight_diff_x = next(
                        k for k, value in enumerate(weight_diff_x) if value > maximum_tick)  # find the end of axis
                for _col in first_weight_diff_df.columns:
                    if numpy.sum(first_weight_diff_df[_col]) == 0:
                        continue
                    weight_diff_axis.plot(weight_diff_x[0:end_weight_diff_x],
                                          first_weight_diff_df[_col].iloc[0:end_weight_diff_x], label=_col, linewidth=2)
                weight_diff_axis.axvline(x=first_herd_effect_delay, color='r', label='herd effect delay')

                weight_diff_axis.grid()
                weight_diff_axis.legend(ncol=4, prop={'size': 8})
                weight_diff_axis.set_title(
                    'Subplot ' + str(folder_index + 1) + 'b - model weight diff: ' + draw_info.titles[folder_index])
                weight_diff_axis.set_xlabel('time (tick)')
                weight_diff_axis.set_ylabel('weight diff')
                weight_diff_axis.set_yscale('log')
                weight_diff_axis.set_xlim([0, maximum_tick])
                # weight_diff_axis.set_xlim([0, final_weight_diff_df.index[end_weight_diff_x]])

            if draw_topology_map:
                topology_axis = whole_axs[current_row * number_of_plot_per_row + 2, current_col]
                topology_axis.set_axis_off()
                G = topology_graphs[folder_index]
                layout = nx.nx_agraph.graphviz_layout(G)
                nx.draw(G, pos=layout, font_color='k', alpha=1.0, linewidths=0.1, width=0.1, font_size=8,
                        ax=topology_axis,
                        node_size=2)
                if draw_topology_map_in_bitmap_format:
                    topology_axis.set_rasterized(True)

        whole_fig.tight_layout()
        whole_fig.savefig(save_name + '.pdf', dpi=200)
        whole_fig.savefig(save_name + '.jpg', dpi=200)
        plt.close(whole_fig)

        #plot the herd effect
        if herd_effect_draw_with_size:
            fig, axs = plt.subplots(3, 1, figsize=(10, 10))
            axs[0].plot(herd_effect_delay_df["size"], herd_effect_delay_df['herd_effect_delay'])
            axs[0].set_xlabel('network size')
            axs[0].set_ylabel('herd effect delay (tick)')
            axs[0].grid()
            # axs[0].set_xlim([herd_effect_delay_df.index.min(), herd_effect_delay_df.index.max()])

            axs[1].plot(herd_effect_delay_df["size"], herd_effect_delay_df['herd_effect_delay'])
            axs[1].set_xlabel('network size(log)')
            axs[1].set_xscale('log')
            axs[1].set_ylabel('herd effect delay (tick)')
            axs[1].grid()
            # axs[1].set_xlim([herd_effect_delay_df.index.min(), herd_effect_delay_df.index.max()])


            def moving_average(data, window_size):
                return data.rolling(window=window_size, min_periods=1).mean()
            smooth_window_size = 15
            herd_effect_delay_df['smoothed_herd_effect_delay'] = moving_average(herd_effect_delay_df['herd_effect_delay'], smooth_window_size)
            axs[2].plot(herd_effect_delay_df["size"], herd_effect_delay_df['smoothed_herd_effect_delay'])
            axs[2].set_xlabel('network size(log)')
            axs[2].set_ylabel('herd effect delay (tick)(smoothed ' + str(smooth_window_size) + ')')
            axs[2].grid()

            fig.savefig('herd_effect_delay.pdf')
            fig.savefig('herd_effect_delay.jpg')
            herd_effect_delay_df.to_csv('herd_effect_delay.csv')
        else:
            fig, axs = plt.subplots(1, 1, figsize=(20, 5))
            axs.plot(herd_effect_delay_df["network_name"], herd_effect_delay_df['herd_effect_delay'])
            axs.set_xlabel('network name')
            axs.set_ylabel('herd effect delay (tick)')
            fig.savefig('herd_effect_delay.pdf')
            fig.savefig('herd_effect_delay.jpg')
            herd_effect_delay_df.to_csv('herd_effect_delay.csv')



    flag_generate_for_each_result = query_yes_no(
        'do you want to draw accuracy graph and weight difference graph for each simulation result?', default="no")
    if flag_generate_for_each_result:
        for folder_index in range(len(draw_info.folders)):
            folder = draw_info.folders[folder_index]
            subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]
            assert len(subfolders) != 0
            for each_test_result_folder in subfolders:
                print("processing: " + each_test_result_folder)
                accuracy_file_path = each_test_result_folder + '/accuracy.csv'
                accuracy_df = pandas.read_csv(accuracy_file_path, index_col=0, header=0)

                weight_diff_file_path = each_test_result_folder + '/model_weight_diff.csv'
                weight_diff_df = pandas.read_csv(weight_diff_file_path, index_col=0, header=0)

                accuracy_x = accuracy_df.index
                accuracy_df_len = len(accuracy_df)

                weight_diff_x = weight_diff_df.index
                weight_diff_df_len = len(weight_diff_df)

                fig, axs = plt.subplots(2, figsize=(10, 10))
                for col in accuracy_df.columns:
                    axs[0].plot(accuracy_x, accuracy_df[col], label=col, alpha=0.75)

                for col in weight_diff_df.columns:
                    axs[1].plot(weight_diff_x, weight_diff_df[col], label=col)

                axs[0].grid()
                axs[0].legend(ncol=5)
                if len(axs[0].columns) > 10:
                    axs[0].legend().remove()
                axs[0].set_title('accuracy')
                axs[0].set_xlabel('time (tick)')
                axs[0].set_ylabel('accuracy (0-1)')
                axs[0].set_xlim([0, accuracy_df.index[accuracy_df_len - 1]])
                axs[0].set_ylim([0, 1])

                axs[1].grid()
                axs[1].legend()
                axs[1].set_title('model weight diff')
                axs[1].set_xlabel('time (tick)')
                axs[1].set_ylabel('weight diff')
                axs[1].set_yscale('log')
                axs[1].set_xlim([0, weight_diff_df.index[weight_diff_df_len - 1]])

                fig.tight_layout()
                fig.savefig(each_test_result_folder + '/accuracy_weight_diff_combine.pdf')
                fig.savefig(each_test_result_folder + '/accuracy_weight_diff_combine.jpg', dpi=800)
                plt.close(fig)



