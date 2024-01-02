import networkx as nx
import pandas
import os
import networkx
import json
import time
import importlib.util
import pickle


def int_to_text(num):
    result = ""
    while num:
        num, remainder = divmod(num - 1, 26)
        result = chr(65 + remainder) + result
    return result


def save_data(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def try_load_data(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
            return obj
    return None


def try_calculate_layout_with_cache(G: nx.Graph, cache_path="layout.pkl"):
    layout_cache = try_load_data(cache_path)
    if layout_cache is None:
        layout_cache = nx.nx_agraph.graphviz_layout(G)
        save_data(layout_cache, cache_path)
    return layout_cache


def calculate_node_size_for_drawing(G: nx.Graph) -> int:
    N = len(G.nodes)
    node_size = int(50000/N)
    node_size = max(10, node_size)
    return node_size


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


def calculate_herd_effect_delay(accuracy_series: pandas.Series, first_average_time=60):
    accuracy_series = accuracy_series.rolling(window=3).mean()
    average_accuracy_diff = accuracy_series.diff()
    average_accuracy_diff.dropna(inplace=True)
    # herd_effect_delay_tick = average_accuracy_diff.idxmax()
    largest_diff = average_accuracy_diff.nlargest(10)
    largest_indexes = largest_diff.index
    for i in largest_indexes:
        if i > first_average_time:
            return i


def graph_centrality_normalized(G, centrality_function=nx.current_flow_betweenness_centrality):
    def sum_of_deviations_from_max(values):
        max_val = max(values)
        return sum(max_val - v for v in values)

    node_centrality = nx.betweenness_centrality(G)
    graph_centrality = sum_of_deviations_from_max(list(node_centrality.values()))
    network_size = len(G.nodes)
    star_network = nx.star_graph(network_size-1)
    star_network_centrality = nx.betweenness_centrality(star_network)
    star_graph_centrality = sum_of_deviations_from_max(list(star_network_centrality.values()))
    graph_centrality = graph_centrality / star_graph_centrality
    return graph_centrality
