import json


def load_simulator_config_file(file_path: str):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def save_simulator_config_file(config_target, file_path: str):
    with open(file_path, 'w') as file:
        json.dump(config_target, file, indent=4)
