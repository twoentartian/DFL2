import data_process_lib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    accuracy_file_path = 'accuracy.csv'
    accuracy_df = data_process_lib.load_csv_with_parquet_acceleration(accuracy_file_path)

    communities = data_process_lib.try_load_data("final_communities.pkl")

    output_df = pd.DataFrame(columns=["node_name", "herd_effect_delay", "community", "community_size"])
    for column in accuracy_df:
        data = accuracy_df[column]
        delay = data_process_lib.calculate_herd_effect_delay(data)
        new_row = pd.DataFrame({'herd_effect_delay': delay, "node_name": column, "community": 0, "community_size":0}, index=[0])
        output_df = pd.concat([output_df.loc[:], new_row]).reset_index(drop=True)
    output_df.to_csv("herd_effect_delay.csv")

    node_community = {}
    for i, community in enumerate(communities):
        for single_node in community:
            matching_row_index = output_df.index[output_df['node_name'] == single_node]
            output_df.at[matching_row_index[0], 'community'] = data_process_lib.int_to_text(i+1)
            output_df.at[matching_row_index[0], 'community_size'] = len(community)

    whole_fig, whole_axs = plt.subplots(1, 1, figsize=(10, 10), squeeze=False)
    axs = whole_axs[0,0]
    axs.scatter(output_df.index, sorted(output_df["herd_effect_delay"]))
    whole_fig.savefig("herd_effect_delay.pdf")

    if community is not None:
        sorted_df = output_df.sort_values(by='community_size', ascending=True)
        whole_fig, whole_axs = plt.subplots(1, 1, figsize=(10, 10), squeeze=False)
        axs = whole_axs[0,0]
        axs.scatter(sorted_df["community_size"], sorted_df["herd_effect_delay"], alpha=0.1, c="C0")
        average_herd_delays = {}
        for index, row in sorted_df.iterrows():
            if average_herd_delays.get(row["community_size"]) is None:
                average_herd_delays[row["community_size"]] = []
            average_herd_delays[row["community_size"]].append(row["herd_effect_delay"])
        for key, values in average_herd_delays.items():
            average_herd_delays[key] = np.average(values)

        lists = sorted(average_herd_delays.items())
        x, y = zip(*lists)
        axs.scatter(x, y, c="C1")
        whole_fig.savefig("herd_effect_delay_community.pdf")