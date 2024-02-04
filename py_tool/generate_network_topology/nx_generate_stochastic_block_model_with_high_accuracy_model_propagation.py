import sys
import networkx as nx
import argparse
import json

import nx_lib


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser(description="generate star networks", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("size_a", help="size of block A", type=int)
    parser.add_argument("size_b", help="size of block B", type=int)
    parser.add_argument("paa", help="edge probability within A", type=float)
    parser.add_argument("pbb", help="edge probability within B", type=float)
    parser.add_argument("pab", help="edge probability between A and B", type=float)

    args = parser.parse_args()
    config = vars(args)
    if len(sys.argv) < 6:
        print("use: python ./nx_generate_stochastic_block_model.py {size_a} {size_b} {paa} {pbb} {pab}")
        exit(1)
    size_a = int(config['size_a'])
    size_b = int(config['size_b'])
    paa = float(config['paa'])
    pbb = float(config['pbb'])
    pab = float(config['pab'])

    print(f"size of block A/B: {size_a}/{size_b}")
    print(f"edge probability of block A/B: {paa}/{pbb}")
    print(f"edge probability between A and B: {pab}")

    sizes = [size_a, size_b]
    probabilities = [[paa, pab], [pab, pbb]]

    # Generate the stochastic block model
    G = nx.stochastic_block_model(sizes, probabilities)

    print(f"total edges: {len(G.edges)}")

    name = f"a{size_a}_b{size_b}_paa{paa}_pbb{pbb}_pab{pab}"
    nx_lib.generate_topology_file(G, name)
    nx_lib.save_network_info(G, name, True)

    # generate stage script files
    block_a = range(0, size_a, 1)
    block_b = range(size_a, size_a+size_b, 1)
    # a to b
    tick_to_enable = size_a * 150
    node_config = {
        "enable": True
    }
    changed_nodes = {}
    for node in block_b:
        changed_nodes[str(node)] = node_config
    script_a_to_b = []
    script_a_to_b.append({
        "tick": tick_to_enable,
        "script": changed_nodes,
    })
    f = open("script_a_to_b.json", "w")
    f.write(json.dumps(script_a_to_b, indent=4))
    f.close()
    # b to a
    tick_to_enable = size_b * 150
    node_config = {
        "enable": True
    }
    changed_nodes = {}
    for node in block_a:
        changed_nodes[str(node)] = node_config
    script_b_to_a = []
    script_b_to_a.append({
        "tick": tick_to_enable,
        "script": changed_nodes,
    })
    f = open("script_b_to_a.json", "w")
    f.write(json.dumps(script_b_to_a, indent=4))
    f.close()

