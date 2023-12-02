import networkx as nx
import pandas
import os
import networkx
import json
import time
import importlib.util
import pickle


def save_data(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def try_load_data(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
            return obj
    return None


def calculate_node_size_for_drawing(G: nx.Graph) -> int:
    N = len(G.nodes)
    node_size = int(50000/N)
    node_size = max(10, node_size)
    return node_size


def graph_centrality(G, vertex_centrality_func):
    def __sum_of_deviations_from_max(values):
        max_val = max(values)
        return sum(max_val - v for v in values)
    vertex_centrality = vertex_centrality_func(G)
    output = __sum_of_deviations_from_max(list(vertex_centrality.values()))
    return output


def graph_centrality_normalized(G, vertex_centrality_func):
    c = graph_centrality(G, vertex_centrality_func)
    star_graph = nx.star_graph(len(G.nodes()))
    c_star = graph_centrality(star_graph, vertex_centrality_func)
    return c/c_star


def load_csv_with_parquet_acceleration(file_path: str, force_load_csv=False) -> pandas.DataFrame:
    spam_spec = importlib.util.find_spec("fastparquet")
    found_fastparquet = spam_spec is not None
    spam_spec = importlib.util.find_spec("pyarrow")
    found_pyarrow = spam_spec is not None

    parquet_path = file_path + ".parquet"
    if os.path.exists(parquet_path) and not force_load_csv and found_fastparquet:
        df = pandas.read_parquet(parquet_path, engine="fastparquet")
        return df
    else:
        df = pandas.read_csv(file_path, index_col=0, header=0, engine="pyarrow" if found_pyarrow else None)
        if found_fastparquet:
            df.to_parquet(parquet_path, engine="fastparquet")
        return df


def load_graph_from_simulation_config(config_file_path: str, verbose=False) -> networkx.Graph:
    t = 0
    if verbose:
        t = time.time()
        print(f"loading simulation config")
    config_file = open(config_file_path)
    config_file_content = config_file.read()
    config_file_json = json.loads(config_file_content)
    topology = config_file_json['node_topology']
    peer_control_enabled = config_file_json['services']['time_based_hierarchy_service']['enable']
    nodes = config_file_json['nodes']

    # DiGraph or Graph?
    G = networkx.Graph()
    for singleItem in topology:
        dirLink = singleItem.split('->')
        if len(dirLink) != 1:
            G = networkx.DiGraph()
            break

    nodes_to_add = []
    for single_node in nodes:
        nodes_to_add.append(single_node['name'])
    G.add_nodes_from(nodes_to_add)

    if not peer_control_enabled:
        edges_to_add = []
        for singleItem in topology:
            unDirLink = singleItem.split('--')
            if len(unDirLink) != 1:
                edges_to_add.append((unDirLink[0], unDirLink[1]))
                edges_to_add.append((unDirLink[1], unDirLink[0]))

            dirLink = singleItem.split('->')
            if len(dirLink) != 1:
                edges_to_add.append((unDirLink[0], unDirLink[1]))
        G.add_edges_from(edges_to_add)
    if verbose:
        print(f"finish loading simulation config, elapsed {time.time()-t}")
    return G


def calculate_herd_effect_delay(accuracy_df: pandas.DataFrame):
    pass

