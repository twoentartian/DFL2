import os
import shutil
import pathlib


def move(source_dir, target_dir):
    simulation_folder = os.listdir(source_dir)
    assert(len(simulation_folder) == 1)
    simulation_folder = simulation_folder[0]
    real_src_dir = os.path.join(source_dir, simulation_folder)
    file_names = os.listdir(real_src_dir)
    pathlib.Path(target_dir).mkdir(parents=True, exist_ok=True)
    for file_name in file_names:
        shutil.move(os.path.join(real_src_dir, file_name), target_dir)
    shutil.rmtree(source_dir)


if __name__ == "__main__":
    current_directory = os.getcwd()
    sub_items = os.listdir(current_directory)
    for item in sub_items:
        current_dir = os.path.join(current_directory, item)
        if not os.path.isdir(current_dir):
            continue
        child_dirs = item.split("___")
        output_path = current_directory
        for i in child_dirs:
            output_path = os.path.join(output_path, i)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
        # pathlib.Path(output_path).mkdir(parents=True, exist_ok=False)
        move(current_dir, output_path)
