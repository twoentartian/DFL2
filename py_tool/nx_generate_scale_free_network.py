import sys
import random
import networkx as nx
import numpy as np
import cmath
import math
import argparse
import copy

import nx_lib

if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser(description="generate scale free networks with modification", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("number_of_node", help="number of nodes", type=int)
    parser.add_argument("average_degree", help="average degree", type=int)

    args = parser.parse_args()
    config = vars(args)
    if len(sys.argv) < 3:
        print("use: python ./nx_generate_scale_free_network.py {number_of_node} {average_degree} {other_args}*")
        exit(1)
    n = int(config['number_of_node'])
    average_degree = int(config['average_degree'])

    print("number of nodes: " + str(n))
    print("average degree: " + str(average_degree))

    total_edges = n * average_degree
    d = (n**2) - (4*total_edges)    # calculate the discriminant, d = (b**2) - (4*a*c)
    m = (n - cmath.sqrt(d))/2       # use the smaller root, sol1 = (-b-cmath.sqrt(d))/(2*a)
    m = m.real
    m_floor = math.floor(m)
    m_ceil = math.ceil(m)
    prob = m - m_floor

    G: nx.Graph = nx.dual_barabasi_albert_graph(n=n, m1=m_floor, m2=m_ceil, p=1-prob, seed=np.random, initial_graph=None)

    # shuffle nodes
    G = nx_lib.shuffle_node(G)

    name = "n" + str(n) + "." + "avg" + str(round(m))
    nx_lib.generate_topology_file(G, name)
    nx_lib.save_network_info(G, name)


