import datetime
import os
import json
from subprocess import call
import shutil
import networkx as nx
import nx_lib
import script_lib
import simulation_config_lib
import sys
import random

repeat = 1
simulation_name = "test_output"

simulation_max_tick = 25000


def generate_topology() -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from([0,1,2,3,4])
    return G


def special_nodes() -> {}:
    special_nodes = {}
    special_nodes["0"] = {"node_type": "normal"}
    special_nodes["1"] = {"node_type": "normal"}
    special_nodes["2"] = {"node_type": "normal"}
    special_nodes["3"] = {"node_type": "normal"}
    special_nodes["4"] = {"node_type": "normal"}
    return special_nodes


def generate_script():
    script_content = []
    script_lib.set_node_type(script_content, 5000, 0, "pontificator")
    script_lib.add_edge_no_direction(script_content, 5000, [(0,1),(0,2),(0,3),(0,4)])

    return script_content


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
        # create output folder
        output_folder_name = f"{simulation_name}_repeat{r}"
        output_folder_dir = os.path.join(current_path, output_folder_name)
        output_network_list.append(output_folder_name)
        if not os.path.exists(output_folder_dir):
            os.mkdir(output_folder_dir)

        # generate topology
        G = generate_topology()
        nx_lib.generate_topology_file(G, os.path.join(tool_folder_path, "temp"))

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

        # generate stage script
        scripts = generate_script()
        script_lib.save_script_to_file(scripts, os.path.join(output_folder_dir, "script.json"))

        # modify simulator config
        simulator_config_json = simulation_config_lib.load_simulator_config_file(
            os.path.join(output_folder_dir, simulator_config_file_name))
        special_node_dict = special_nodes()
        for node_name, node_prop in special_node_dict.items():
            for k, v in node_prop.items():
                # get node
                for n in simulator_config_json["nodes"]:
                    if n["name"] == node_name:
                        n[k] = v
                        break

        simulator_config_json["ml_max_tick"] = simulation_max_tick
        simulator_config_json["services"]["stage_manager"]["enable"] = True
        simulator_config_json["services"]["stage_manager"]["script_path"] = "./script.json"
        with open(sys.argv[0], mode="r") as src_file:
            simulator_config_json["comment_this_config_file_is_generated_by_py_script"] = src_file.read()
        simulation_config_lib.save_simulator_config_file(simulator_config_json,
                                                         os.path.join(output_folder_dir, simulator_config_file_name))

        # copy run script and simulator to output folder
        shutil.copyfile(os.path.join(simulation_folder_path, simulator_name),
                        os.path.join(output_folder_dir, simulator_name))
        shutil.copyfile(os.path.join(simulation_folder_path, simulator_script_name),
                        os.path.join(output_folder_dir, simulator_script_name))

    # generator run script
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    file_name = f"simulation_list_{current_time}.json"
    with open(file_name, 'w') as file:
        list_file_json = {'list_file_json': output_network_list,
                          'run_simulator_command': f'sh ./{simulator_script_name}'}
        file.write(json.dumps(list_file_json, indent=4))


