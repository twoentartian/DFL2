import networkx as nx
import numpy as np
import random
import multiprocessing


def sum_of_deviations_from_max(values):
    max_val = max(values)
    return sum(max_val - v for v in values)


def graph_centrality(G, vertex_centrality_func):
    vertex_centrality = vertex_centrality_func(G)
    output = sum_of_deviations_from_max(list(vertex_centrality.values()))
    return output


def test_case(index, N=100, vertex_centrality_func=nx.degree_centrality):
    p0 = random.uniform(0.05, 1)
    p1 = random.uniform(0.05, 1)
    G0 = nx.fast_gnp_random_graph(N, p0)
    G1 = nx.fast_gnp_random_graph(N, p1)
    while not nx.is_connected(G0):
        G0 = nx.fast_gnp_random_graph(N, p0)
    while not nx.is_connected(G1):
        G1 = nx.fast_gnp_random_graph(N, p1)

    between_G0 = graph_centrality(G0, vertex_centrality_func)
    between_G1 = graph_centrality(G1, vertex_centrality_func)
    G_union = nx.compose(G0, G1)
    between_Gunion = graph_centrality(G_union, vertex_centrality_func)
    if between_Gunion <= np.maximum(between_G0, between_G1):
        print(f"#{index} {between_Gunion} <= max({between_G0},{between_G1})    {vertex_centrality_func}", flush=True)
    else:
        print(f"#{index} ERROR! {between_Gunion} > max({between_G0},{between_G1})", flush=True)
        print("press any keys to continue", flush=True)
        input()


if __name__ == "__main__":
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as executor:
        arg = range(1, 10000)
        executor.map(test_case, arg)

        # nx.degree_centrality #counter case found
        # nx.eigenvector_centrality # no counter case found, n=100, repeat =10000
        # nx.betweenness_centrality # no counter case found, n=100, repeat =10000
        # nx.closeness_centrality #counter case found
        # nx.degree_centrality #counter case found
        # nx.current_flow_betweenness_centrality # no counter case found, n=100, repeat =10000