import datetime
import os
import json
from subprocess import call
import shutil
import networkx as nx
import random
import numpy as np
import nx_lib

# network_size = [100, 200, 400, 800, 1600, 2400, 3200, 4800]
network_size = 500
star_ranges = range(50, 501, 25)
repeat = 10

network_generator = "large_scale_simulation_generator_maksim"
non_iid_generator = "dirichlet_distribution_config_generator"
generate_non_iid = False
non_iid_generator_arg = "5"

simulator_config_file_name = "simulator_config.json"
network_generator_config_file_name = "large_scale_config_maksim.json"

simulator_name = "DFL_simulator_opti"
simulator_script_name = "run_simulator_opti.sh"

simulation_folder_name = "simulation"
tool_folder_name = "tool"

current_path = os.getcwd()
tool_folder_path = os.path.join(current_path, tool_folder_name)
simulation_folder_path = os.path.join(current_path, simulation_folder_name)

network_generator_path = os.path.join(tool_folder_path, network_generator)
non_iid_generator = os.path.join(tool_folder_path, non_iid_generator)

simulator_config_path = os.path.join(simulation_folder_path, simulator_config_file_name)
network_generator_config_path = os.path.join(tool_folder_path, network_generator_config_file_name)


if __name__ == "__main__":
    output_network_list = []

    # check simulation config exist
    if not os.path.exists(simulator_config_path):
        print("simulator config file not exists")
        exit(-1)

    if not os.path.exists(network_generator_config_path):
        print("network generator config file not exists")
        exit(-1)

    for r in range(repeat):
        for star_range in star_ranges:
            # create output folder
            output_folder_name = f"n{network_size}_star{star_range}_repeat{r}"
            output_folder_dir = os.path.join(current_path, output_folder_name)
            output_network_list.append(output_folder_name)
            if not os.path.exists(output_folder_dir):
                os.mkdir(output_folder_dir)

            # generate topology
            G_base: nx.Graph = nx_lib.generate_random_network(network_size, 3)
            G_base = nx_lib.shuffle_node(G_base)
            G_star: nx.Graph = nx_lib.generate_star_network(star_range, 1)
            G_output = nx_lib.combine_two_networks(G_base, G_star)
            nx_lib.generate_topology_file(G_output, os.path.join(tool_folder_path, "temp"))

            # update network size
            f = open(network_generator_config_path, "r")
            config_json = json.load(f)
            config_json["network_topology_maksim_format"] = "./temp.data"
            f.close()
            f = open(network_generator_config_path, "w")
            output_json_data = json.dumps(config_json, indent=4)
            f.write(output_json_data)
            f.close()

            # generate simulator config
            status = call(network_generator_path, cwd=tool_folder_path, shell=True)
            assert status == 0

            # generate non_iid distribution
            if generate_non_iid:
                status = call(non_iid_generator + " " + non_iid_generator_arg, cwd=tool_folder_path, shell=True)
                assert status == 0

            # copy simulator config to output folder
            shutil.copyfile(simulator_config_path, os.path.join(output_folder_dir, simulator_config_file_name))

            # copy run script and simulator to output folder
            shutil.copyfile(os.path.join(simulation_folder_path, simulator_name), os.path.join(output_folder_dir, simulator_name))
            shutil.copyfile(os.path.join(simulation_folder_path, simulator_script_name), os.path.join(output_folder_dir, simulator_script_name))

    # generator run script
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    file_name = f"simulation_list_{current_time}.json"
    with open(file_name, 'w') as file:
        list_file_json = {'list_file_json': output_network_list,
                          'run_simulator_command': f'sh ./{simulator_script_name}'}
        file.write(json.dumps(list_file_json, indent=4))


