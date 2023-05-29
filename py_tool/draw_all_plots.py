import os
import sys

import numpy
import pandas
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

row = 3
col = 3

# folders = ["50_node_8_random_peers", "50_node_grid", "50_node_loop", "50_node_star"]
folders = ["100_node", "200_node", "400_node", "800_node", "1600_node", "2400_node", "3200_node", "4000_node", "4800_node"]
# titles = ["random network - 8 peers", "grid", "loop", "star"]
titles = ["100_node", "200_node", "400_node", "800_node", "1600_node", "2400_node", "3200_node", "4000_node", "4800_node"]

maximum_tick = 10000
save_name = "draw"
draw_model_weight_diff = True
draw_topology_map = False

reduce_image_file_size = True
lines_limit_per_image = 50

herd_effect_weight_diff_reference_layer = "conv2"


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

    # calculate the average of accuracy graph
    min_accuracy: pandas.Series = arg_accuracy_df.min(axis=1)
    min_accuracy_averaged = min_accuracy.rolling(window=7).mean()
    herd_effect_delay_tick = min_accuracy_averaged[min_accuracy_averaged > 0.1].first_valid_index()
    return herd_effect_delay_tick


if __name__ == "__main__":
    assert len(folders) <= row * col
    folder_names_set = set()
    for folder_index in range(len(folders)):
        folder = folders[folder_index]
        assert not (folder in folder_names_set)
        folder_names_set.add(folder)

    flag_generate_whole = query_yes_no('do you want to generate the whole figure?')
    if flag_generate_whole:
        figsize_col = 5 * col
        figsize_row = 2.5 * row
        plot_row = row
        plot_col = col
        number_of_plot_per_row = 1
        if draw_model_weight_diff:
            figsize_row = figsize_row + figsize_row
            plot_row = plot_row + row
            number_of_plot_per_row = number_of_plot_per_row + 1
        if draw_topology_map:
            figsize_row = figsize_row + figsize_row
            plot_row = plot_row + row
            number_of_plot_per_row = number_of_plot_per_row + 1

        whole_fig, whole_axs = plt.subplots(plot_row, plot_col, figsize=(figsize_col, figsize_row), squeeze=False)
        topology_graphs = []
        herd_effect_delay_df = pandas.DataFrame(columns=['herd_effect_delay'])
        for folder_index in range(len(folders)):
            current_col = folder_index % col
            current_row = folder_index // col

            folder = folders[folder_index]

            subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]
            assert len(subfolders) != 0
            final_accuracy_df = pandas.DataFrame()
            final_weight_diff_df = pandas.DataFrame()
            is_first_dataframe = True
            first_to_draw_topology = True

            for each_test_result_folder in subfolders:
                # load the topology
                if first_to_draw_topology:
                    import json
                    import networkx as nx

                    first_to_draw_topology = False
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
                    topology_graphs.append(G)

                accuracy_file_path = each_test_result_folder + '/accuracy.csv'
                accuracy_df = pandas.read_csv(accuracy_file_path, index_col=0, header=0)
                # print(accuracy_df)

                weight_diff_file_path = each_test_result_folder + '/model_weight_diff.csv'
                weight_diff_df = pandas.read_csv(weight_diff_file_path, index_col=0, header=0)
                # print(weight_diff_df)

                if is_first_dataframe:
                    is_first_dataframe = False
                    final_accuracy_df = accuracy_df
                    final_weight_diff_df = weight_diff_df
                else:
                    final_accuracy_df = final_accuracy_df.add(accuracy_df, fill_value=0)
                    final_weight_diff_df = final_weight_diff_df.add(weight_diff_df, fill_value=0)
            final_accuracy_df = final_accuracy_df.div(len(subfolders))
            final_weight_diff_df = final_weight_diff_df.div(len(subfolders))
            print(final_accuracy_df)
            print(final_weight_diff_df)

            # calculate herd effect delay
            herd_effect_delay = calculate_herd_effect_delay(final_accuracy_df, final_weight_diff_df)
            simulation_name = folders[folder_index]
            herd_effect_delay_df.loc[simulation_name] = {'herd_effect_delay': herd_effect_delay}
            print("herd effect delay = " + str(herd_effect_delay))

            accuracy_x = final_accuracy_df.index
            accuracy_df_len = len(final_accuracy_df)
            weight_diff_x = final_weight_diff_df.index
            weight_diff_df_len = len(final_weight_diff_df)

            accuracy_axis = whole_axs[current_row * number_of_plot_per_row, current_col]

            if maximum_tick >= accuracy_x[len(accuracy_x) - 1]:
                end_accuracy_x = len(accuracy_x) - 1
            else:
                end_accuracy_x = next(
                    k for k, value in enumerate(accuracy_x) if value > maximum_tick)  # find the end of axis

            if not reduce_image_file_size:
                # normal plot
                for _col in final_accuracy_df.columns:
                    accuracy_axis.plot(accuracy_x[0:end_accuracy_x], final_accuracy_df[_col].iloc[0:end_accuracy_x], label=_col, alpha=0.75)
            else:
                if lines_limit_per_image < len(final_accuracy_df.columns):
                    ratio = 1 / len(final_accuracy_df.columns)
                    target_ratio_per_draw = 1 / lines_limit_per_image
                    current_ratio = 0
                    for _col in final_accuracy_df.columns:
                        if current_ratio > target_ratio_per_draw:
                            accuracy_axis.plot(accuracy_x[0:end_accuracy_x], final_accuracy_df[_col].iloc[0:end_accuracy_x], label=_col, alpha=0.75)
                            current_ratio -= target_ratio_per_draw
                        current_ratio += ratio

                else:
                    for _col in final_accuracy_df.columns:
                        accuracy_axis.plot(accuracy_x[0:end_accuracy_x], final_accuracy_df[_col].iloc[0:end_accuracy_x], label=_col, alpha=0.75)
            accuracy_axis.axvline(x=herd_effect_delay, color='r', label='herd effect delay')

            accuracy_axis.grid()
            accuracy_axis.legend(ncol=5)
            if len(final_accuracy_df.columns) > 10:
                accuracy_axis.legend().remove()
            accuracy_axis.set_title('Subplot ' + str(folder_index + 1) + 'a - accuracy: ' + titles[folder_index])
            accuracy_axis.set_xlabel('time (tick), herd_effect=' + str(herd_effect_delay) + " #node=" + str(len(final_accuracy_df.columns)))
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
                for _col in final_weight_diff_df.columns:
                    if numpy.sum(final_weight_diff_df[_col]) == 0:
                        continue
                    weight_diff_axis.plot(weight_diff_x[0:end_weight_diff_x],
                                          final_weight_diff_df[_col].iloc[0:end_weight_diff_x], label=_col, linewidth=2)
                weight_diff_axis.axvline(x=herd_effect_delay, color='r', label='herd effect delay')

                weight_diff_axis.grid()
                weight_diff_axis.legend(ncol=4, prop={'size': 8})
                weight_diff_axis.set_title(
                    'Subplot ' + str(folder_index + 1) + 'b - model weight diff: ' + titles[folder_index])
                weight_diff_axis.set_xlabel('time (tick), herd_effect=' + str(herd_effect_delay) + " #node=" + str(len(final_accuracy_df.columns)))
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

        whole_fig.tight_layout()
        whole_fig.savefig(save_name + '.pdf')
        whole_fig.savefig(save_name + '.jpg', dpi=800)
        plt.close(whole_fig)

        #plot the herd effect
        fig, axs = plt.subplots(1, 1)
        axs.plot(herd_effect_delay_df.index, herd_effect_delay_df['herd_effect_delay'])
        axs.set_xlabel('network name')
        axs.set_ylabel('herd effect delay (tick)')
        fig.savefig('herd_effect_delay.pdf')
        fig.savefig('herd_effect_delay.jpg', dpi=800)


    flag_generate_for_each_result = query_yes_no(
        'do you want to draw accuracy graph and weight difference graph for each simulation result?', default="no")
    if flag_generate_for_each_result:
        for folder_index in range(len(folders)):
            folder = folders[folder_index]
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
                axs[0].set_xlabel('time (tick), herd_effect_delay=' + str())
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



