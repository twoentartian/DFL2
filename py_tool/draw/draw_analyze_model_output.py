import os
import pandas
import matplotlib.pyplot as plt
import zipfile
import math
import shutil
import re

analyze_models_output_path = "analyze_models_output"

accuracy_detail_file_name = 'accuracy_detail.zip'
weight_distance_from_each_other_file_name = 'weight_distance_from_each_other.zip'
weight_distance_from_start_point_origin_file_name = 'weight_distance.zip'

number_of_figure_per_row = 5
maximum_tick = 20000

for items in os.listdir(analyze_models_output_path):
    if items.count('.zip') == 0:
        continue

    item_name = items.replace('.zip', '')
    zip_file_path = os.path.join(analyze_models_output_path, items)
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(analyze_models_output_path)
    cache_path = os.path.join(analyze_models_output_path, item_name)

    if items == accuracy_detail_file_name:                      # accuracy_detail.zip
        accuracy_detail_files = os.listdir(cache_path)
        accuracy_detail_file_paths = {}
        for single_accuracy_detail_file in accuracy_detail_files:
            node_index = int(single_accuracy_detail_file.replace('.csv', ''))
            accuracy_detail_file_paths[node_index] = os.path.join(cache_path, single_accuracy_detail_file)
        accuracy_detail_file_paths = dict(sorted(accuracy_detail_file_paths.items()))
        total_number_of_nodes = len(accuracy_detail_file_paths)

        col_count = number_of_figure_per_row
        row_count = math.ceil(total_number_of_nodes / col_count)
        whole_fig, whole_axis = plt.subplots(row_count, col_count, figsize=(col_count * 5, row_count * 2.5))
        file_path_list = list(accuracy_detail_file_paths.keys())
        for file_index in range(total_number_of_nodes):
            current_col = file_index % col_count
            current_row = file_index // col_count
            file = accuracy_detail_file_paths[file_path_list[file_index]]
            accuracy_df = pandas.read_csv(file, index_col=0, header=0)
            accuracy_df.sort_index(axis='index', inplace=True)
            print(accuracy_df)
            current_axis = whole_axis[current_row, current_col]
            accuracy_x = accuracy_df.index
            accuracy_df_len = len(accuracy_df)
            if maximum_tick >= accuracy_x[len(accuracy_x) - 1]:
                end_accuracy_x = len(accuracy_x) - 1
            else:
                end_accuracy_x = next(k for k, value in enumerate(accuracy_x) if value > maximum_tick)  # find the end of axis

            for _col in accuracy_df.columns:
                current_axis.plot(accuracy_x[0:end_accuracy_x], accuracy_df[_col].iloc[0:end_accuracy_x], label=_col, alpha=0.75)
            current_axis.grid()
            current_axis.legend(ncol=5, fontsize=5)
            if len(accuracy_df.columns) > 50:
                current_axis.legend().remove()
            current_axis.set_title('Accuracy details for node: ' + str(file_path_list[file_index]))
            current_axis.set_xlabel('time (tick)')
            current_axis.set_ylabel('accuracy (0-1)')
            current_axis.set_xlim([0, accuracy_df.index[end_accuracy_x]])
            current_axis.set_ylim([0, 1])
        whole_fig.tight_layout()
        whole_fig.savefig(item_name + '.pdf')
        # whole_fig.savefig(item_name + '.jpg', dpi=800)
        plt.close(whole_fig)

    if items == weight_distance_from_each_other_file_name:      # weight_distance_from_each_other.zip
        distance_files = os.listdir(cache_path)
        distance_file_paths_start = {}
        all_node_names = set()
        for single_file in distance_files:
            match = re.match(r"(\d+)-(\d+).csv", single_file)
            if match:
                node_small, node_large = int(match.group(1)), int(match.group(2))
                all_node_names.add(node_small)
                all_node_names.add(node_large)
                if node_large < node_small:
                    temp = node_small
                    node_small = node_large
                    node_large = node_small
                distance_file_paths_start[str(node_small) + "-" + str(node_large)] = os.path.join(cache_path, single_file)

        output_path_str = "weight_distance_from_each_other_output"
        if not os.path.exists(output_path_str):
            os.mkdir(output_path_str)

        for node_0 in all_node_names:
            save_path = os.path.join(output_path_str, str(node_0) + '.pdf')
            if os.path.exists(save_path):
                print("skip draw mutual weight distance for node %x" % node_0)
                continue

            col_count = number_of_figure_per_row
            row_count = math.ceil(len(all_node_names) / col_count)
            whole_fig, whole_axis = plt.subplots(row_count, col_count, figsize=(col_count * 5, row_count * 2.5))
            axis_index = -1
            for node_1 in all_node_names:
                axis_index = axis_index + 1
                if node_0 == node_1:
                    continue
                print(str(node_0) + " to node " + str(node_1))
                current_col = axis_index % col_count
                current_row = axis_index // col_count
                current_axis = whole_axis[current_row, current_col]

                node_small = node_0
                node_large = node_1
                if node_large < node_small:
                    temp = node_small
                    node_small = node_large
                    node_large = temp
                csv_file_name_0 = str(node_small) + "-" + str(node_large) + ".csv"
                csv_file_name_1 = str(node_large) + "-" + str(node_small) + ".csv"
                csv_file_path_0 = os.path.join(cache_path, csv_file_name_0)
                csv_file_path_1 = os.path.join(cache_path, csv_file_name_1)
                if os.path.exists(csv_file_path_0):
                    data = pandas.read_csv(csv_file_path_0, index_col=0, header=0)
                elif os.path.exists(csv_file_path_1):
                    data = pandas.read_csv(csv_file_path_1, index_col=0, header=0)
                else:
                    raise AssertionError("the file to describe distance between node %x and node %x is missing" % (node_0, node_1))

                data_x = data.index
                data_len = len(data)
                if maximum_tick >= data_x[len(data_x) - 1]:
                    max_tick = len(data_x) - 1
                else:
                    max_tick = next(k for k, value in enumerate(data) if value > maximum_tick)  # find the end of axis

                for _col in data.columns:
                    current_axis.plot(data_x[0:max_tick], data[_col].iloc[0:max_tick], label=_col, alpha=0.75)

                current_axis.grid()
                current_axis.legend(ncol=5, fontsize=5)
                if len(data.columns) > 10:
                    current_axis.legend().remove()
                current_axis.set_title(str(node_small) + "-" + str(node_large) + " weight distance")
                current_axis.set_xlabel('time (tick)')
                current_axis.set_ylabel('distance')
                current_axis.set_yscale('log')
                current_axis.set_xlim([0, data.index[max_tick]])
            whole_fig.tight_layout()
            print("save figure for node " + str(node_0))
            whole_fig.savefig(save_path)
            plt.close("all")

    if items == weight_distance_from_start_point_origin_file_name:      # weight_distance_from_start_point_origin.zip
        distance_files = os.listdir(cache_path)
        distance_file_paths_start = {}
        distance_file_paths_origin = {}
        distance_file_paths_destination = {}
        distance_file_paths_delta = {}
        all_node_names = set()
        for single_file in distance_files:
            match = re.search(r"\d+", single_file)
            if match:
                node_name = int(match.group())
            else:
                raise AssertionError("no node name in %s"%single_file)
            all_node_names.add(node_name)

            data_type = ''
            node_description = single_file.replace('.csv', '')
            if node_description.count('start'):
                distance_file_paths_start[node_name] = os.path.join(cache_path, single_file)
            if node_description.count('origin'):
                distance_file_paths_origin[node_name] = os.path.join(cache_path, single_file)
            if node_description.count('destination'):
                distance_file_paths_destination[node_name] = os.path.join(cache_path, single_file)
            if node_description.count('delta'):
                distance_file_paths_delta[node_name] = os.path.join(cache_path, single_file)

        all_node_names_index = 0
        node_name_to_index = {}
        for name in all_node_names:
            node_name_to_index[name] = all_node_names_index
            all_node_names_index = all_node_names_index + 1

        col_count = number_of_figure_per_row
        row_count = math.ceil(len(all_node_names) / col_count)
        whole_fig_start, whole_axis_start = plt.subplots(row_count, col_count, figsize=(col_count * 5, row_count * 2.5))
        whole_fig_origin, whole_axis_origin = plt.subplots(row_count, col_count, figsize=(col_count * 5, row_count * 2.5))
        whole_fig_destination, whole_axis_destination = plt.subplots(row_count, col_count, figsize=(col_count * 5, row_count * 2.5))
        whole_fig_start_log, whole_axis_start_log = plt.subplots(row_count, col_count, figsize=(col_count * 5, row_count * 2.5))
        whole_fig_origin_log, whole_axis_origin_log = plt.subplots(row_count, col_count, figsize=(col_count * 5, row_count * 2.5))
        whole_fig_destination_log, whole_axis_destination_log = plt.subplots(row_count, col_count, figsize=(col_count * 5, row_count * 2.5))
        whole_fig_delta, whole_axis_delta = plt.subplots(row_count, col_count, figsize=(col_count * 5, row_count * 2.5))
        whole_fig_delta_log, whole_axis_delta_log = plt.subplots(row_count, col_count, figsize=(col_count * 5, row_count * 2.5))

        def draw_distance(distance_file_paths, whole_axis, _type, use_log_y=True):
            for [_node_name, path] in distance_file_paths.items():
                print("process %s" % path)

                _data = pandas.read_csv(path, index_col=0, header=0)
                _node_index = node_name_to_index[_node_name]

                _current_col = _node_index % col_count
                _current_row = _node_index // col_count
                _current_axis = whole_axis[_current_row, _current_col]

                _data_x = _data.index
                _data_len = len(_data)
                if maximum_tick >= _data_x[len(_data_x) - 1]:
                    max_tick = len(_data_x) - 1
                else:
                    max_tick = next(k for k, value in enumerate(_data) if value > maximum_tick)  # find the end of axis

                for _col in _data.columns:
                    _current_axis.plot(_data_x[0:max_tick], _data[_col].iloc[0:max_tick], label=_col, alpha=0.75)

                _current_axis.grid()
                _current_axis.legend(ncol=5, fontsize=5)
                if len(_data.columns) > 10:
                    _current_axis.legend().remove()
                _current_axis.set_title("node %d distance to %s" % (_node_name, _type))
                _current_axis.set_xlabel('time (tick)')
                _current_axis.set_ylabel('distance')
                if use_log_y:
                    _current_axis.set_yscale('log')
                _current_axis.set_xlim([0, _data.index[max_tick]])


        draw_distance(distance_file_paths_start, whole_axis_start, 'start', use_log_y=False)
        print("save figure distance_to_start.pdf")
        whole_fig_start.tight_layout()
        whole_fig_start.savefig('distance_to_start.pdf')

        draw_distance(distance_file_paths_origin, whole_axis_origin, 'origin', use_log_y=False)
        print("save figure distance_to_origin.pdf")
        whole_fig_origin.tight_layout()
        whole_fig_origin.savefig('distance_to_origin.pdf')

        draw_distance(distance_file_paths_destination, whole_axis_destination, 'destination', use_log_y=False)
        print("save figure distance_to_destination.pdf")
        whole_fig_destination.tight_layout()
        whole_fig_destination.savefig('distance_to_destination.pdf')

        draw_distance(distance_file_paths_start, whole_axis_start_log, 'start', use_log_y=True)
        print("save figure distance_to_start_log.pdf")
        whole_fig_start_log.tight_layout()
        whole_fig_start_log.savefig('distance_to_start_log.pdf')

        draw_distance(distance_file_paths_origin, whole_axis_origin_log, 'origin', use_log_y=True)
        print("save figure distance_to_origin_log.pdf")
        whole_fig_origin_log.tight_layout()
        whole_fig_origin_log.savefig('distance_to_origin_log.pdf')

        draw_distance(distance_file_paths_destination, whole_axis_destination_log, 'destination', use_log_y=True)
        print("save figure distance_to_destination_log.pdf")
        whole_fig_destination_log.tight_layout()
        whole_fig_destination_log.savefig('distance_to_destination_log.pdf')

        draw_distance(distance_file_paths_delta, whole_axis_delta, 'previous(delta_weight_change)', use_log_y=False)
        print("save figure delta_weight_distance.pdf")
        whole_fig_delta.tight_layout()
        whole_fig_delta.savefig('delta_weight_distance.pdf')

        draw_distance(distance_file_paths_delta, whole_axis_delta_log, 'previous(delta_weight_change)', use_log_y=True)
        print("save figure delta_weight_distance_log.pdf")
        whole_fig_delta_log.tight_layout()
        whole_fig_delta_log.savefig('delta_weight_distance_log.pdf')

    shutil.rmtree(cache_path, ignore_errors=True)
