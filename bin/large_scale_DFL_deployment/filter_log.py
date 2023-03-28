import os
import shutil

output_dir_name = "filter_log"

if __name__ == "__main__":
    output_dir = os.path.join(os.curdir, output_dir_name)
    if os.path.exists(output_dir):
        os.removedirs(output_dir)
    os.mkdir(output_dir)

    for single_entry in os.listdir(os.curdir):
        print("processing " + single_entry)
        if not os.path.isdir(single_entry):
            continue    # not a dir
        if single_entry.startswith("node"):
            # this is a node
            src_dir = os.path.join(os.curdir, single_entry)
            output_dir = os.path.join(os.curdir, output_dir_name, single_entry)
            os.mkdir(output_dir)
            output_log_dir = os.path.join(output_dir, "log")
            os.mkdir(output_log_dir)
            shutil.copy(os.path.join(src_dir, "node_log.txt"), os.path.join(output_dir, "node_log.txt"))
            shutil.copy(os.path.join(src_dir, "log", "DFL.INFO"), os.path.join(output_log_dir, "DFL.INFO"))
