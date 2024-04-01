import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np
from pathlib import Path

layer_names = ['conv1', 'conv2', 'ip1', 'ip2']
# layer_names = ['conv1']


def load_csv(csv_file_name: str):
    csv_path = csv_file_name
    print(f"Loading csv file {csv_path}")
    previous_time = time.time()
    df = pd.read_csv(f"{csv_path}", low_memory=False, engine="c")
    print(f"Loading csv file {csv_path} finished - {(time.time() - previous_time):.2f} seconds")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="draw weight distribution from delta weight file",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("csv_file_name", help="the path to delta weight csv file", type=str)

    args = parser.parse_args()
    config = vars(args)
    csv_path = config["csv_file_name"]

    if not os.path.exists(csv_path):
        print(f"{csv_path} does not exist, exit...")
        exit(0)

    df = load_csv(csv_path)
    weight_distribution_path = "weight_distribution"
    for layer_name in layer_names:
        Path(f"./{weight_distribution_path}/{layer_name}").mkdir(parents=True, exist_ok=True)
        filtered_df = df.filter(regex=f"({layer_name}-\d+|type|tick)$")
        all_weights = {}
        current_tick = -1
        for index, row in filtered_df.iterrows():
            if row["type"] == "init":
                for column_name, value in row.items():
                    if layer_name in column_name:
                        all_weights[column_name] = value
                # plot
                weight_values = list(all_weights.values())
                whole_fig, whole_axs = plt.subplots(1, 1, figsize=(12, 6), squeeze=False)
                whole_axs[0, 0].hist(weight_values, bins='auto')
                whole_fig.savefig(f"./{weight_distribution_path}/{layer_name}/{layer_name}-init.png")
                plt.close(whole_fig)
            if row["type"] == "train":
                for column_name, value in row.items():
                    if layer_name in column_name:
                        all_weights[column_name] += value
            if row["type"] == "average":
                for column_name, value in row.items():
                    if layer_name in column_name:
                        all_weights[column_name] += value
            tick_now = row["tick"]
            if current_tick < tick_now:
                current_tick = tick_now
                # plot
                weight_values = list(all_weights.values())
                whole_fig, whole_axs = plt.subplots(1, 1, figsize=(12, 6), squeeze=False)
                whole_axs[0, 0].hist(weight_values, bins='auto')
                whole_fig.savefig(f"./{weight_distribution_path}/{layer_name}/{layer_name}-{tick_now}.png")
                plt.close(whole_fig)
