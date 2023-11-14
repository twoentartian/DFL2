import sys
import argparse
import json
import nx_lib
import os
import copy
from subprocess import call
import concurrent.futures
import threading

output_folder_name = "lower_bound_graph_generated"
current_path = os.getcwd()
output_folder_path = os.path.join(current_path, output_folder_name)

# worker = os.cpu_count()
worker = 1


def get_lower_bound_graph(simulation_folder, low_bound_k):
    simulation_folder = str(simulation_folder)
    G = nx_lib.get_graph_from_dfl_simulation_config(simulation_folder)
    G_original = copy.deepcopy(G)

    # remove edges until we arrive the Regular(k=3) as close as possible
    print(f"processing {simulation_folder}, edge={len(G.edges)}")
    all_nodes = sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True)
    for single_node in all_nodes:
        neighbors = list(G.neighbors(single_node))
        sorted_neighbors = sorted(neighbors, key=lambda x: G.degree(x), reverse=True)
        for single_neighbor in sorted_neighbors:
            if G.degree[single_node] > low_bound_k and G.degree[single_neighbor] > low_bound_k:
                G.remove_edge(single_node, single_neighbor)

    # add the largest hub
    node_with_largest_degree = max(G_original, key=G_original.degree)
    edges_of_largest_node = list(G_original.edges(node_with_largest_degree))
    G.add_edges_from(edges_of_largest_node)

    if worker == 1:
        nx_lib.save_network_info(G, os.path.join(output_folder_path, simulation_folder), enable_topology=True)
        nx_lib.save_network_info(G_original, os.path.join(output_folder_path, "original_" + simulation_folder), enable_topology=True)
    nx_lib.generate_topology_file(G, os.path.join(output_folder_path, simulation_folder))
    nx_lib.generate_topology_file(G_original, os.path.join(output_folder_path, "original_" + simulation_folder))
    print(f"finish processing {simulation_folder}, edge={len(G.edges)}")


if __name__ == "__main__":
    if not os.path.exists(output_folder_path):
        os.mkdir(output_folder_path)

    parser = argparse.ArgumentParser(description="calculate the lower bound graph with network list file", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("list_file", help="the path to network list file", type=str)
    parser.add_argument('--min_k', nargs='?', const=3, default=3, type=int)

    args = parser.parse_args()
    config = vars(args)

    if len(sys.argv) < 2:
        print("use: python ./nx_generate_lower_bound_graph_from_simulator_config_files.py {network_list_file_path}/{simulator_config.json}")
        exit(1)

    list_file_path = config['list_file']
    min_k = int(config['min_k'])
    basename = os.path.basename(list_file_path)
    if basename == "simulator_config.json":
        get_lower_bound_graph(basename, min_k)
    else:
        list_file_json = ''
        with open(list_file_path) as list_file:
            list_file_content = list_file.read()
            list_file_json = json.loads(list_file_content)
        simu_configs = list_file_json['list_file_json']
        min_k_s = [min_k for i in range(len(simu_configs))]
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker) as executor:
            executor.map(get_lower_bound_graph, simu_configs, min_k_s)
