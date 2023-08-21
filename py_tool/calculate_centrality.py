import os
import json
import networkx as nx
import pandas as pd

import draw_info


def sum_of_deviations_from_max(values):
    max_val = max(values)
    return sum(max_val - v for v in values)


if __name__ == "__main__":
    assert len(draw_info.folders) <= draw_info.row * draw_info.col
    folder_names_set = set()
    output_df = pd.DataFrame(columns=['simulation_config', 'simulation_case', 'centrality'])
    for folder_index in range(len(draw_info.folders)):
        folder = draw_info.folders[folder_index]
        print(f"processing {folder}")
        assert not (folder in folder_names_set)
        folder_names_set.add(folder)
        current_col = folder_index % draw_info.col
        current_row = folder_index // draw_info.col

        subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]
        assert len(subfolders) != 0
        for each_test_result_folder in subfolders:
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

            node_centrality = nx.current_flow_betweenness_centrality(G)
            graph_centrality = sum_of_deviations_from_max(list(node_centrality.values()))

            # https://igraph.org/c/doc/igraph-Structural.html#igraph_centralization
            network_size = len(G.nodes)
            star_network = nx.star_graph(network_size-1)
            star_network_centrality = nx.current_flow_betweenness_centrality(star_network)
            star_graph_centrality = sum_of_deviations_from_max(list(star_network_centrality.values()))
            graph_centrality = graph_centrality / star_graph_centrality
            f = open(each_test_result_folder + "/centrality.txt", "w")
            f.write(f"{graph_centrality}")
            f.close()

            output_df.loc[len(output_df)] = [folder, each_test_result_folder, graph_centrality]

    output_df.to_csv("centrality.csv")


