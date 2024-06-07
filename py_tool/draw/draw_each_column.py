import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np


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
    axis.set_ylabel('column value')
    axis.set_xlim([0, max(df['tick'])])


def plot(df: pd.DataFrame, output_file_name: str):
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
    whole_fig, whole_axs = plt.subplots(2, 1, figsize=(12, 12), squeeze=False)
    for col in df.columns:
        if col == 'tick':
            continue
        data = df[col]
        whole_axs[0, 0].plot(df['tick'], data, alpha=0.5, linewidth=1)
        whole_axs[1, 0].plot(df['tick'], data, alpha=0.5, linewidth=1)
    whole_axs[0, 0].set_yscale('linear')
    whole_axs[1, 0].set_yscale('log')
    axis_apply_style(whole_axs[0, 0], f"{output_file_name}", df)
    axis_apply_style(whole_axs[1, 0], f"{output_file_name}", df)
    whole_fig.tight_layout()
    whole_fig.savefig(f"{output_file_name}.pdf")
    plt.close(whole_fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="draw each column in a csv file",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("csv_file_name", help="the path to a csv file", type=str)

    args = parser.parse_args()
    config = vars(args)
    csv_path = config["csv_file_name"]
    if not os.path.exists(csv_path):
        print(f"{csv_path} does not exist, exit...")
        exit(0)

    df = load_csv(csv_path)
    csv_file_name = os.path.basename(csv_path)

    plot(df, csv_file_name)
