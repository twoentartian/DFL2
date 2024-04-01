import networkx as nx
import matplotlib.pyplot as plt
import data_process_lib
import argparse
import numpy as np
import pandas as pd

config_file_path = 'simulator_config.json'
accuracy_file_path = 'accuracy.csv'

conservativeness_level = 0.5
propagating_starting_tick = 2000
fusion_accuracy_mapping_file_path = "fusion_model_accuracy.csv"
training_per_tick = 10

video_cache_path = "./video_cache"


def round_to_nearest(num, divisor):
    return round(num / divisor) * divisor


def calculate_transformation_matrix(G: nx.Graph, conservativeness_level: float, propagating_src):
    network_size = len(G.nodes)
    adjacent_matrix = nx.adjacency_matrix(G)
    adjacent_matrix = adjacent_matrix.todense()
    node_mapping = {node: i for i, node in enumerate(G.nodes())}
    transformation_matrix = np.zeros((network_size, network_size))
    for single_node in G.nodes:
        single_node_index = node_mapping[single_node]
        if single_node in propagating_src:
            transformation_matrix[single_node_index, single_node_index] = 1
        else:
            non_zero_count = np.count_nonzero(adjacent_matrix[single_node_index])
            self_weight = conservativeness_level
            whole_other_weight = 1 - conservativeness_level
            single_other_weight = whole_other_weight / non_zero_count
            transformation_matrix[single_node_index] = adjacent_matrix[single_node_index] * single_other_weight
            transformation_matrix[single_node_index, single_node_index] = self_weight
    return transformation_matrix


def get_propagating_src(fusion_accuracy_mapping):
    propagating_src = [col for col in fusion_accuracy_mapping.columns if 'accuracy' not in col and 'loss' not in col]
    return propagating_src


def calculate_fusion_accuracy(transformation_matrix, fusion_accuracy_mapping, tick_to_draw):
    tick_step = tick_to_draw[1] - tick_to_draw[0]
    tick_start = tick_to_draw[0]
    assert tick_step >= training_per_tick
    transformation_matrix_tick_step = np.linalg.matrix_power(transformation_matrix, int(round(tick_step/training_per_tick)))
    output_fusion_accuracy = pd.DataFrame(index=tick_to_draw, columns=[i for i in range(len(transformation_matrix[0]))])
    output_fusion_loss = pd.DataFrame(index=tick_to_draw, columns=[i for i in range(len(transformation_matrix[0]))])
    propagating_src = get_propagating_src(fusion_accuracy_mapping)
    fusion_accuracy_precision = fusion_accuracy_mapping[propagating_src[0]].drop_duplicates().nsmallest(2)
    fusion_accuracy_precision = fusion_accuracy_precision.values
    fusion_accuracy_precision = fusion_accuracy_precision[1] - fusion_accuracy_precision[0]
    for single_tick in tick_to_draw:
        print(f"processing fusion accuracy/loss for tick {single_tick}")
        P = (single_tick - tick_start) / tick_step
        P = int(round(P))
        transformation_matrix_now = np.linalg.matrix_power(transformation_matrix_tick_step, P)
        # transformation_matrix_now = np.linalg.matrix_power(transformation_matrix, (single_tick - tick_start)//training_per_tick)
        for index, each_row in enumerate(transformation_matrix_now):
            fusion_value = {v: round_to_nearest(each_row[int(v)], fusion_accuracy_precision) for v in propagating_src}
            mask = True
            for key, value in fusion_value.items():
                mask = mask & (abs(fusion_accuracy_mapping[key] - value) < 0.0001)
            select_row = fusion_accuracy_mapping[mask]
            output_fusion_accuracy.at[single_tick, index] = select_row['accuracy'].values[0]
            output_fusion_loss.at[single_tick, index] = select_row['loss'].values[0]

    return output_fusion_accuracy, output_fusion_loss


if __name__ == "__main__":
    # parser args
    parser = argparse.ArgumentParser(description="generate a video for accuracy trends, put this script with \"simulator_config.json\"", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--override_cache", help="override images cache?", type=bool)
    args = parser.parse_args()
    config = vars(args)
    override_cache = False
    if config["override_cache"]:
        override_cache = True

    G = data_process_lib.load_graph_from_simulation_config(config_file_path)

    accuracy_df = pd.read_csv(accuracy_file_path, index_col="tick")

    # tick_to_draw
    total_tick = len(accuracy_df.index)
    draw_counter = 0
    tick_to_draw = []
    for tick in accuracy_df.index:
        tick_to_draw.append(tick)

    # transformation_matrix
    fusion_accuracy_mapping = pd.read_csv(fusion_accuracy_mapping_file_path)
    propagating_src = get_propagating_src(fusion_accuracy_mapping)
    transformation_matrix = calculate_transformation_matrix(G, conservativeness_level, propagating_src)
    tick_to_draw_fusion = [num for num in tick_to_draw if num >= propagating_starting_tick]
    fusion_accuracy, fusion_loss = calculate_fusion_accuracy(transformation_matrix, fusion_accuracy_mapping, tick_to_draw_fusion)
    fusion_accuracy.to_csv("fusion_accuracy.csv")
    fusion_loss.to_csv("fusion_loss.csv")
