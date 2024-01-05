import os
import shutil
from random import random

# Define the source and destination directories
destination_dirs = [f"n500_star{i}" for i in range(50,501,25)]


# Move the contents of each source directory to the destination
for destination_dir in destination_dirs:
    if not os.path.exists(destination_dir):
        os.mkdir(destination_dir)
    source_dirs = [f"{destination_dir}_repeat{i}" for i in range(0, 5)]
    for src_dir in source_dirs:
        if os.path.exists(src_dir):
            for item in os.listdir(src_dir):
                src_path = os.path.join(src_dir, item)
                dst_path = os.path.join(destination_dir, item)
                if os.path.exists(dst_path):
                    dst_path = dst_path + str(int(random()*100))
                if os.path.isdir(src_path):
                    # Use shutil.move to move folders
                    shutil.move(src_path, dst_path)
                    shutil.rmtree(src_dir)

print("Folders have been moved.")
