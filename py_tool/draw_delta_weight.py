import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np

layer_names = ['conv1', 'conv2', 'ip1', 'ip2']


def axis_apply_style(axis, draw_name, df, train_or_average):
    axis.grid(which='major', color='#DDDDDD', linewidth=0.8)
    axis.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
    axis.set_title(f"{draw_name}-{train_or_average}")
    axis.set_xlabel('tick')
    axis.set_ylabel('delta model weight')
    axis.set_yscale('symlog', linthresh=0.0001)
    axis.set_xlim([0, max(df['tick'])])


def load_csv(csv_file_name: str):
    csv_path = csv_file_name
    print(f"Loading csv file {csv_path}")
    previous_time = time.time()
    df = pd.read_csv(f"{csv_path}", low_memory=False, engine="c")
    print(f"Loading csv file {csv_path} finished - {(time.time() - previous_time):.2f} seconds")
    return df


def plot_weight_n_order_derivative(df: pd.DataFrame, save_name: str, n_derivative: int, train_df=None, average_df=None,
                                   desired_lines_per_figure=200, layer_to_draw=layer_names,
                                   n_order_derivative_output_folder='order_derivative'):
    output_folder = f"{n_derivative}_{n_order_derivative_output_folder}"

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    item_counter = {}
    for column in df:
        if '-' in column:
            items = column.split('-')
            if len(items) == 2:
                if item_counter.get(items[0]) is None:
                    item_counter[items[0]] = 0
                else:
                    item_counter[items[0]] = item_counter[items[0]] + 1

    # train and average df
    if train_df is None:
        train_df = df[df["type"] == "train"]
    if average_df is None:
        average_df = df[df["type"] == "average"]

    # draw
    for layer_name, counter in item_counter.items():
        total_figures = counter // desired_lines_per_figure + 1
        for figure_index in range(total_figures):
            start_line_index = figure_index * desired_lines_per_figure
            end_line_index = (figure_index + 1) * desired_lines_per_figure
            end_line_index = min(end_line_index, counter)

            whole_fig, whole_axs = plt.subplots(2, 1, figsize=(12, 12), squeeze=False)
            print(f"processing: {layer_name}: {start_line_index}-{end_line_index}")
            for line_index in range(start_line_index, end_line_index):
                column = f"{layer_name}-{line_index}"
                temp_train = train_df[column]
                temp_average = average_df[column]
                for n in range(n_derivative):
                    temp_train = np.gradient(temp_train)
                    temp_average = np.gradient(temp_average)
                whole_axs[0, 0].plot(train_df['tick'], temp_train, color='r', alpha=0.1, linewidth=0.01)
                whole_axs[1, 0].plot(average_df['tick'], temp_average, color='r', alpha=0.1, linewidth=0.01)
            draw_name = f"{save_name}-{layer_name}-{start_line_index}-{end_line_index}"
            axis_apply_style(whole_axs[0, 0], draw_name, df, "train")
            axis_apply_style(whole_axs[1, 0], draw_name, df, "average")
            whole_axs[0, 0].set_ylim(whole_axs[1, 0].get_ylim())
            whole_fig.tight_layout()
            whole_fig.savefig(f"{output_folder}/{draw_name}.pdf")
            plt.close(whole_fig)


def plot_weight_filtered(df: pd.DataFrame, save_name: str, train_df=None, average_df=None,
                         desired_lines_per_figure=500, layer_to_draw=layer_names,
                         plot_weight_output_folder_filtered='delta_weight_filtered'):
    if not os.path.exists(plot_weight_output_folder_filtered):
        os.mkdir(plot_weight_output_folder_filtered)
    # import matplotlib as mpl
    # mpl.rcParams['path.simplify'] = True

    item_counter = {}
    for column in df:
        if '-' in column:
            items = column.split('-')
            if len(items) == 2:
                if item_counter.get(items[0]) is None:
                    item_counter[items[0]] = 0
                else:
                    item_counter[items[0]] = item_counter[items[0]] + 1

    # calculate when to draw
    item_perform_plot_threshold = {}
    for k, v in item_counter.items():
        item_perform_plot_threshold[k] = int(v / desired_lines_per_figure)

    # train and average df
    if train_df is None:
        train_df = df[df["type"] == "train"]
    if average_df is None:
        average_df = df[df["type"] == "average"]

    # draw
    all_figs = []
    all_axs = []
    for i in range(len(layer_to_draw)):
        whole_fig, whole_axs = plt.subplots(2, 1, figsize=(12, 12), squeeze=False)
        all_figs.append(whole_fig)
        all_axs.append(whole_axs)

    item_perform_plot_count = {}
    for k, v in item_counter.items():
        item_perform_plot_count[k] = 0
    for column in train_df:
        if '-' in column:
            items = column.split('-')
            if len(items) != 2:
                continue
            layer_name = items[0]
            if item_perform_plot_threshold[layer_name] != item_perform_plot_count[layer_name]:
                item_perform_plot_count[layer_name] = item_perform_plot_count[layer_name] + 1
            else:
                item_perform_plot_count[layer_name] = 0
                for i in range(len(layer_to_draw)):
                    layer_name = layer_to_draw[i]
                    if items[0] == layer_name:
                        all_axs[i][0, 0].plot(train_df['tick'], train_df[column], color='r', alpha=0.1, linewidth=0.01)
    item_perform_plot_count = {}
    for k, v in item_counter.items():
        item_perform_plot_count[k] = 0
    for column in average_df:
        if '-' in column:
            items = column.split('-')
            if len(items) != 2:
                continue
            layer_name = items[0]
            if item_perform_plot_threshold[layer_name] != item_perform_plot_count[layer_name]:
                item_perform_plot_count[layer_name] = item_perform_plot_count[layer_name] + 1
            else:
                item_perform_plot_count[layer_name] = 0
                for i in range(len(layer_to_draw)):
                    layer_name = layer_to_draw[i]
                    if items[0] == layer_name:
                        all_axs[i][1, 0].plot(average_df['tick'], average_df[column], color='r', alpha=0.1,
                                              linewidth=0.01)

    for i in range(len(layer_to_draw)):
        draw_name = f"{save_name}-{layer_to_draw[i]}"
        axis = all_axs[i]

        axis_apply_style(axis[0, 0], draw_name, df, "train")
        axis_apply_style(axis[1, 0], draw_name, df, "average")

        fig = all_figs[i]
        fig.tight_layout()
        fig.savefig(f"{plot_weight_output_folder_filtered}/{draw_name}.pdf")
        plt.close(fig)


def plot_delta_distance(df: pd.DataFrame, save_name: str, train_df=None, average_df=None, layer_to_draw=layer_names,
                        plot_distance='delta_distance'):
    if not os.path.exists(plot_distance):
        os.mkdir(plot_distance)

    item_counter = set()
    for column in df:
        if '+' in column:
            items = column.split('+')
            if len(items) == 2:
                layer_name = items[1]
                if not layer_name in item_counter:
                    item_counter.add(layer_name)
                else:
                    raise Exception(f"distance for layer {layer_name} already exists")

    # train and average df
    if train_df is None:
        train_df = df[df["type"] == "train"]
    if average_df is None:
        average_df = df[df["type"] == "average"]

    # draw
    for column in df:
        if '+' in column:
            items = column.split('+')
            if len(items) == 2 and items[0] == "distance":
                whole_fig, whole_axs = plt.subplots(2, 1, figsize=(12, 12), squeeze=False)
                layer_name = items[1]
                print(f"processing distance: {layer_name}")
                whole_axs[0, 0].plot(train_df['tick'], train_df[column], color='r', alpha=1, linewidth=2)
                whole_axs[1, 0].plot(average_df['tick'], average_df[column], color='r', alpha=1, linewidth=2)
                draw_name = f"{save_name}-{layer_name}"
                axis_apply_style(whole_axs[0, 0], draw_name, df, "train")
                axis_apply_style(whole_axs[1, 0], draw_name, df, "average")
                whole_axs[0, 0].set_ylim(whole_axs[1, 0].get_ylim())
                whole_fig.tight_layout()
                whole_fig.savefig(f"{plot_distance}/{draw_name}.pdf")
                plt.close(whole_fig)


def plot_delta_angle(df: pd.DataFrame, save_name: str, train_df=None, average_df=None, desired_lines_per_figure=200,
                     layer_to_draw=layer_names, plot_angle='delta_angle'):
    if not os.path.exists(plot_angle):
        os.mkdir(plot_angle)

    item_counter = {}
    for column in df:
        if '-' in column:
            items = column.split('-')
            if len(items) == 3:
                if item_counter.get(items[0]) is None:
                    item_counter[items[0]] = 0
                else:
                    item_counter[items[0]] = item_counter[items[0]] + 1

    # train and average df
    if train_df is None:
        train_df = df[df["type"] == "train"]
    if average_df is None:
        average_df = df[df["type"] == "average"]

    # draw
    for layer_name, counter in item_counter.items():
        total_figures = counter // desired_lines_per_figure + 1
        for figure_index in range(total_figures):
            start_line_index = figure_index * desired_lines_per_figure
            end_line_index = (figure_index + 1) * desired_lines_per_figure
            end_line_index = min(end_line_index, counter)

            whole_fig, whole_axs = plt.subplots(2, 1, figsize=(12, 12), squeeze=False)
            print(f"processing: {layer_name}: {start_line_index}-{end_line_index}")
            for line_index in range(start_line_index, end_line_index):
                column = f"{layer_name}-{line_index}-angle"
                whole_axs[0, 0].plot(train_df['tick'], train_df[column], color='r', alpha=0.1, linewidth=0.01)
                whole_axs[1, 0].plot(average_df['tick'], average_df[column], color='r', alpha=0.1, linewidth=0.01)
            draw_name = f"{save_name}-{layer_name}-{start_line_index}-{end_line_index}"
            axis_apply_style(whole_axs[0, 0], draw_name, df, "train")
            axis_apply_style(whole_axs[1, 0], draw_name, df, "average")
            whole_axs[0, 0].set_ylim(whole_axs[1, 0].get_ylim())
            whole_fig.tight_layout()
            whole_fig.savefig(f"{plot_angle}/{draw_name}.pdf")
            plt.close(whole_fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="draw all delta_weight_plots",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("csv_file_name", help="the path to delta weight csv file", type=str)
    parser.add_argument("--distance", help="calculate distance", action=argparse.BooleanOptionalAction)
    parser.add_argument("--angle", help="calculate angle", action=argparse.BooleanOptionalAction)
    parser.add_argument("--filtered", help="calculate angle", action=argparse.BooleanOptionalAction)
    parser.add_argument("--n0", help="calculate angle", action=argparse.BooleanOptionalAction)
    parser.add_argument("--n1", help="calculate angle", action=argparse.BooleanOptionalAction)
    parser.add_argument("--n2", help="calculate angle", action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    config = vars(args)
    csv_path = config["csv_file_name"]

    if not os.path.exists(csv_path):
        print(f"{csv_path} does not exist, exit...")
        exit(0)

    df = load_csv(csv_path)
    train_df = df[df["type"] == "train"]
    average_df = df[df["type"] == "average"]
    csv_file_name = os.path.basename(csv_path)
    if not (not config["distance"] is None and config["distance"] == False):
        plot_delta_distance(df, csv_file_name, train_df=train_df, average_df=average_df)
    if not (not config["angle"] is None and config["angle"] == False):
        plot_delta_angle(df, csv_file_name, train_df=train_df, average_df=average_df)
    if not (not config["filtered"] is None and config["filtered"] == False):
        plot_weight_filtered(df, csv_file_name, train_df=train_df, average_df=average_df)
    if not (not config["n0"] is None and config["n0"] == False):
        plot_weight_n_order_derivative(df, csv_file_name, 0, train_df=train_df, average_df=average_df)
    if not (not config["n1"] is None and config["n1"] == False):
        plot_weight_n_order_derivative(df, csv_file_name, 1, train_df=train_df, average_df=average_df)
    if not (not config["n2"] is None and config["n2"] == False):
        plot_weight_n_order_derivative(df, csv_file_name, 2, train_df=train_df, average_df=average_df)

