import os
import json
import networkx as nx
import matplotlib.pyplot as plt

import draw_info

if __name__ == "__main__":
    assert len(draw_info.folders) <= draw_info.row * draw_info.col
    folder_names_set = set()
    for folder_index in range(len(draw_info.folders)):
        folder = draw_info.folders[folder_index]
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

            eigenvector_centrality = nx.eigenvector_centrality(G)
            norm_eigenvector_centrality = [len(G.nodes) * v for v in eigenvector_centrality.values()]
            plt.figure(figsize=(10, 7))
            nx.draw(G,
                    with_labels=True,
                    node_color=list(eigenvector_centrality.values()),
                    node_size=norm_eigenvector_centrality,
                    cmap=plt.cm.Reds,
                    )
            plt.show()
            input()