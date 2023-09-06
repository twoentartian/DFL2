import nx_lib
import networkx as nx


if __name__ == "__main__":
    network_sizes = range(20, 1010, 20)
    degree = 3
    for network_size in network_sizes:
        G_base: nx.Graph = nx_lib.generate_random_network(network_size, degree)
        G_base = nx_lib.shuffle_node(G_base)
        # G_star: nx.Graph = nx_lib.generate_star_network(network_size, 1)
        # G_output = nx_lib.combine_two_networks(G_base, G_star)

        name = "n" + str(network_size)
        nx_lib.generate_topology_file(G_base, name)
        nx_lib.save_network_info(G_base, name, True)
