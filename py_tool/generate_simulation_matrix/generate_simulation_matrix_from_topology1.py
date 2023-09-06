import os
import json
from subprocess import call
import shutil

sizes = range(100,1001,50)

topology_files = ["n1000" + ".star" + str(num) + ".data" for num in sizes]

network_generator = "large_scale_simulation_generator_maksim"
non_iid_generator = "dirichlet_distribution_config_generator"
generate_non_iid = False
non_iid_generator_arg = "0.5"

simulator_name = "DFL_simulator_opti"
simulator_script_name = "run_simulator_opti.sh"

simulator_config_file_name = "simulator_config.json"
network_generator_config_file_name = "large_scale_config_maksim.json"

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
    # check simulation config exist
    if not os.path.exists(simulator_config_path):
        print("simulator config file not exists")
        exit(-1)

    if not os.path.exists(network_generator_config_path):
        print("network generator config file not exists")
        exit(-1)

    for topology_file in topology_files:
        # create output folder
        output_folder_dir = os.path.join(current_path, topology_file)
        if not os.path.exists(output_folder_dir):
            os.mkdir(output_folder_dir)

        # update network size
        f = open(network_generator_config_path, "r")
        config_json = json.load(f)
        config_json["network_topology_maksim_format"] = "./maksim/" + topology_file
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
    run_script_content = """
from subprocess import call

$network_size$

run_simulator_command = "sh ./$simulator_script_name$"

if __name__ == "__main__":
    for single_network_size in network_size:
        folder = str(single_network_size)

        status = call(run_simulator_command, cwd=folder, shell=True)
        assert status == 0
    """
    run_script_content = run_script_content.replace("$simulator_script_name$", simulator_script_name)
    network_size_array_elements = ", ".join("\"" + str(i) + "\"" for i in topology_files)
    network_size_line = f"network_size = [{network_size_array_elements}]"
    run_script_content = run_script_content.replace("$network_size$", network_size_line)
    with open("run_simulation_matrix_from_topology_1.py", "w") as run_script:
        run_script.write(run_script_content)

