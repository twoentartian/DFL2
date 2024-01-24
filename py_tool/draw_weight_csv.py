import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np

layer_names = ['conv1', 'conv2', 'ip1', 'ip2']


def load_csv(csv_file_name: str):
    csv_path = csv_file_name
    print(f"Loading csv file {csv_path}")
    previous_time = time.time()
    df = pd.read_csv(f"{csv_path}", low_memory=False, engine="c")
    print(f"Loading csv file {csv_path} finished - {(time.time() - previous_time):.2f} seconds")
    return df


def axis_apply_style(axis, draw_name, df):
    axis.grid(which='major', color='#DDDDDD', linewidth=1)
    axis.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=1)
    axis.set_title(f"{draw_name}")
    axis.set_xlabel('tick')
    axis.set_ylabel('model weight')
    axis.set_xlim([0, max(df['tick'])])


def plot_weight_n_order_derivative(df: pd.DataFrame, csv_file_name: str, n_derivative: int,
                                   desired_lines_per_figure=50,
                                   n_order_derivative_output_folder='order_derivative'):
    output_folder = f"{csv_file_name}_{n_derivative}_{n_order_derivative_output_folder}"

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
                data = df[column]
                for n in range(n_derivative):
                    data = np.gradient(data)
                whole_axs[0, 0].plot(df['tick'], data, alpha=0.5, linewidth=1)
                whole_axs[1, 0].plot(df['tick'], data, alpha=0.5, linewidth=1)
            whole_axs[0, 0].set_yscale('symlog', linthresh=0.0001)
            whole_axs[1, 0].set_yscale('linear')
            draw_name = f"{layer_name}-{start_line_index}-{end_line_index}"
            axis_apply_style(whole_axs[0, 0], f"{csv_file_name}-{draw_name}", df)
            axis_apply_style(whole_axs[1, 0], f"{csv_file_name}-{draw_name}", df)
            whole_fig.tight_layout()
            whole_fig.savefig(f"{output_folder}/{draw_name}.pdf")
            plt.close(whole_fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="draw all delta_weight_plots",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("csv_file_name", help="the path to delta weight csv file", type=str)

    args = parser.parse_args()
    config = vars(args)
    csv_path = config["csv_file_name"]
    if not os.path.exists(csv_path):
        print(f"{csv_path} does not exist, exit...")
        exit(0)

    df = load_csv(csv_path)
    csv_file_name = os.path.basename(csv_path)

    plot_weight_n_order_derivative(df, csv_file_name, 0)
