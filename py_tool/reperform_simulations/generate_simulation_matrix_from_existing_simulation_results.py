import datetime
import os
import json
from subprocess import call
import shutil
from pathlib import Path


simulator_path = r"/home/tyd/temp/DFL_simulator_opti"
simulator_script_path = r"/home/tyd/temp/run_simulator_opti.sh"

simulator_path = r"\\wsl.localhost\Ubuntu\home\tyd\temp\DFL_simulator_opti"
simulator_script_path = r"\\wsl.localhost\Ubuntu\home\tyd\temp\run_simulator_opti.sh"
output_folder_name = "output"


if __name__ == "__main__":
    output_network_list = []

    simulator_script_name = os.path.basename(simulator_script_path)

    if not os.path.exists(simulator_path):
        print(f"simulator path {simulator_path} does not exist")
        exit(-1)
    if not os.path.exists(simulator_script_path):
        print(f"simulator script path {simulator_script_path} does not exist")
        exit(-1)

    current_directory = os.getcwd()
    output_path = os.path.join(current_directory, output_folder_name)
    sub_items = os.listdir(current_directory)
    sub_folders = [item for item in sub_items if (os.path.isdir(os.path.join(current_directory, item)) and (item != "output") )]

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    for sub_folder in sub_folders:
        subsub_items = os.listdir(os.path.join(current_directory, sub_folder))
        subsub_folders = [item for item in subsub_items if os.path.isdir(os.path.join(current_directory, sub_folder, item))]
        for subsub_folder in subsub_folders:
            current_path = os.path.join(current_directory, sub_folder, subsub_folder)
            current_name = f"{sub_folder}___{subsub_folder}"
            current_output_path = os.path.join(output_path, current_name)
            simulator_config_path = os.path.join(current_path, 'simulator_config.json')

            if not os.path.exists(current_output_path):
                os.mkdir(current_output_path)
            shutil.copy(simulator_config_path, current_output_path)
            shutil.copy(simulator_path, current_output_path)
            shutil.copy(simulator_script_path, current_output_path)

            output_network_list.append(current_name)

    # generator run script
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    file_name = f"simulation_list_{current_time}.json"
    with open(os.path.join(output_path, file_name), 'w') as file:
        list_file_json = {'list_file_json': output_network_list,
                          'run_simulator_command': f'sh ./{simulator_script_name}'}
        file.write(json.dumps(list_file_json, indent=4))
