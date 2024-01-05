import os

def delete_files_in_layer1_subfolders(root_folder, file_name):
    subfolders = [os.path.join(root_folder, folder) for folder in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, folder))]
    for folder in subfolders:
        file_path = os.path.join(folder, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")


if __name__ == "__main__":
    current_directory = os.getcwd()

    delete_files_in_layer1_subfolders(current_directory, "simulator_config.json")
    delete_files_in_layer1_subfolders(current_directory, "run_simulator_opti.sh")
    delete_files_in_layer1_subfolders(current_directory, "DFL_simulator_opti")

