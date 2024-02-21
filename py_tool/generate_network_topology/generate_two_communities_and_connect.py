import sys
import networkx as nx
import argparse
import json
import random

import nx_lib


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser(description="generate two communities by providing degree and interconnect edge counts", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("size_a", help="size of block A", type=int)
    parser.add_argument("size_b", help="size of block B", type=int)
    parser.add_argument("ka", help="degree for community A", type=int)
    parser.add_argument("kb", help="degree for community B", type=int)
    parser.add_argument("edge_ab", help="edges between A and B", type=int)
    parser.add_argument("tick_to_enable", help="tick to connect A and B", type=int)

    args = parser.parse_args()
    config = vars(args)
    if len(sys.argv) < 6:
        print("use: python ./generate_two_communities_and_connect.py {size_a} {size_b} {ka} {kb} {edge_ab} {tick_to_enable}")
        exit(1)

    size_a = int(config['size_a'])
    size_b = int(config['size_b'])
    ka = int(config['ka'])
    kb = int(config['kb'])
    edge_ab = int(config['edge_ab'])
    tick_to_enable = int(config['tick_to_enable'])

    # create two communities
    N1 = nx.random_regular_graph(ka, size_a)
    N2 = nx.random_regular_graph(kb, size_b)
    mapping = {}
    for i in range(size_b):
        mapping[i] = i+size_a
    nx.relabel_nodes(N2, mapping, copy=False)
    N = nx_lib.combine_two_networks(N1, N2)

    # add edge
    all_possible_edge = []
    for i in range(size_a):
        for j in range(size_a, size_b+size_a):
            all_possible_edge.append((i, j))
    assert(edge_ab <= len(all_possible_edge))
    plan_edge_between_ab = random.sample(all_possible_edge, edge_ab)
    N.add_edges_from(plan_edge_between_ab)

    name = f"a{size_a}_b{size_b}_ka{ka}_kb{kb}_edgeab{edge_ab}_tick{tick_to_enable}"
    nx_lib.save_network_info(N, name, enable_topology=True)
    nx_lib.generate_topology_file(N, name)

    # generate stage script files
    script_events = []
    block_a = range(0, size_a, 1)
    block_b = range(size_a, size_a+size_b, 1)
    edges_between_a_and_b = []
    for node in N.nodes:
        if not node in block_a:
            continue
        for peer in list(N.neighbors(node)):
            if peer in block_b:
                edges_between_a_and_b.append(f"{node}--{peer}")
    script_events.append({
        "tick": 0,
        "script": {
            "remove_edge": edges_between_a_and_b
        },
    })
    script_events.append({
        "tick": tick_to_enable,
        "script": {
            "add_edge": edges_between_a_and_b,
        },
    })
    f = open("script.json", "w")
    f.write(json.dumps(script_events, indent=4))
    f.close()

