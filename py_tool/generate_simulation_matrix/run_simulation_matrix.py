import sys
import argparse
import json
from subprocess import call
import concurrent.futures
import threading
from datetime import datetime

import simulation_device_config as simulation_config

simulator_command = ''
available_task_affinity = []
lock = threading.Lock()
thread_name_map_to_affinity = {}


def run_one_simulation(simulation_folder):
    global available_task_affinity, thread_name_map_to_affinity
    thread_name = threading.current_thread().name
    with lock:
        simulation_folder = str(simulation_folder)
        if thread_name_map_to_affinity.get(thread_name) is None:
            thread_name_map_to_affinity[thread_name] = available_task_affinity.pop(0)

    affinity = thread_name_map_to_affinity[thread_name]
    command = f'taskset -c {affinity} {simulator_command}'
    current_date_time = datetime.now()
    print(f"{current_date_time} === running {simulation_folder}, thread_id={thread_name} affinity={affinity} command={command}")
    with open(f'{simulation_folder}.log', 'w') as f:
        _status = call(command, cwd=simulation_folder, shell=True, stdout=f, stderr=f)
    print(f"finish running {simulation_folder} with status {_status}")
    assert _status == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run simulation matrix with network list file", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("list_file", help="the path to network list file", type=str)

    args = parser.parse_args()
    config = vars(args)

    if len(sys.argv) < 2:
        print("use: python ./run_simulation_matrix.py {network_list_file_path}")
        exit(1)

    list_file_path = config['list_file']
    list_file_json = ''
    with open(list_file_path) as list_file:
        list_file_content = list_file.read()
        list_file_json = json.loads(list_file_content)
    simulations_to_run = list_file_json['list_file_json']
    simulator_command = list_file_json['run_simulator_command']

    for cpu_index in range(len(simulation_config.cpu_count)):
        this_task_affinity = simulation_config.cpu_count[cpu_index]
        for i in range(simulation_config.simulation_per_core):
            available_task_affinity.append(this_task_affinity)

    print(f"ThreadPoolExecutor size={len(available_task_affinity)}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(available_task_affinity)) as executor:
        executor.map(run_one_simulation, simulations_to_run)
