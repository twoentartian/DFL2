import os
import json
import signal
import subprocess
import datetime

if __name__ == "__main__":
    all_procs_can_be_killed = []
    all_procs_cannot_be_killed = []

    # find summary.json
    with open("summary.json", "rb") as file:
        data = file.read()
        summary = json.loads(data)

    # introducer
    print("starting introducers")
    for single_introducer in summary["introducers"]:
        working_dir = os.path.join(os.curdir, single_introducer["folder"])
        with open(os.path.join(working_dir, "introducer_log.txt"), "w") as log:
            proc = subprocess.Popen(["./DFL_introducer"], cwd=working_dir, stdout=log, stderr=log)
            all_procs_can_be_killed.append(proc)
        print("starting " + single_introducer["name"])
    # input("press Enter to start DFL nodes...")

    # nodes
    print("starting DFL nodes")
    for single_node in summary["nodes"]:
        working_dir = os.path.join(os.curdir, single_node["folder"])
        with open(os.path.join(working_dir, "node_log.txt"), "w") as log:
            dfl_env = os.environ.copy()
            dfl_env["OPENBLAS_NUM_THREADS"] = "1"
            dfl_env["GOTO_NUM_THREADS"] = "1"
            dfl_env["OMP_NUM_THREADS"] = "1"
            proc = subprocess.Popen(["./DFL"], cwd=working_dir, stdout=log, stderr=log, stdin=subprocess.PIPE, shell=True, env=dfl_env)
            all_procs_cannot_be_killed.append((single_node["name"], proc))
        print("starting DFL node: " + single_node["name"])
    # input("press Enter to start data injectors...")

    # data injectors
    print("starting data injectors")
    for single_node in summary["nodes"]:
        working_dir = os.path.join(os.curdir, single_node["folder"])
        with open(os.path.join(working_dir, "injector_log.txt"), "w") as log:
            proc = subprocess.Popen(["./data_injector_mnist"], cwd=working_dir, stdout=log, stderr=log, stdin=subprocess.PIPE, shell=True)
            all_procs_can_be_killed.append(proc)
        print("starting data injector: " + single_node["name"])
    # input("press Enter to wait for DFL nodes exit...")

    # wait for exit
    node_exits = set()
    while True:
        all_exit = True
        for (name, proc) in all_procs_cannot_be_killed:
            try:
                proc.wait(1)
                if name not in node_exits:
                    node_exits.add(name)
                    print("[WARNING] node " + name + " exits at " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            except subprocess.TimeoutExpired:
                all_exit = False
                continue
        if all_exit:
            break

    # kill procs
    print("killing introducer and data injectors")
    for i in all_procs_can_be_killed:
        print("killing " + str(i.pid))
        i.send_signal(signal.SIGINT)
