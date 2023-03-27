import pandas
import matplotlib
import re
import json
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import multiprocessing

log_file_name = "./node_log.txt"
glog_file_name = "./log/DFL.INFO"
temp_file_path = "./temp/"
skip_if_temp_file_exist = True

maximum_time_second = 16000

def is_log(msg: str) -> (bool, str, datetime):
    pattern_glog_header = re.compile(r'[IW]\d{8} \d{2}:\d{2}:\d{2}')

    # glog head
    glog_head = pattern_glog_header.findall(msg)
    if not glog_head:
        return False, None, None
    glog_head = glog_head[0]
    if glog_head[0] == 'I':
        log_level = 'info'
    elif glog_head[0] == 'W':
        log_level = 'warn'
    else:
        log_level = 'unknown'

    # time
    pattern_time = re.compile(r'\d{2}:\d{2}:\d{2}')
    time_str = pattern_time.findall(glog_head)
    if not time_str:
        raise AssertionError('time not available in this glog message: ' + msg)
    log_time = datetime.strptime(time_str[0], '%H:%M:%S')

    return True, log_level, log_time


def get_accuracy(msg: str) -> (bool, float):
    pattern_accuracy = re.compile(r'accuracy: \d\.\d{2}')

    # accuracy
    accuracy_str = pattern_accuracy.findall(msg)
    if not accuracy_str:
        return False, None
    accuracy_str = accuracy_str[0]
    accuracy_str = accuracy_str[10:]
    return True, float(accuracy_str)


def process_single_node(single_node):
    if skip_if_temp_file_exist:
        if os.path.exists(temp_file_path + single_node["name"] + ".csv"):
            return

    print("checking node: " + single_node["name"])
    working_dir = os.path.join(os.curdir, single_node["folder"])
    log = open(os.path.join(working_dir, log_file_name), "r")
    df = pandas.DataFrame(columns=["event_type", "elapsed", "time", "start_time", "node", "accuracy"])

    first_log = True
    day_counter = 0
    last_hour = 0

    # get start date time
    glog_log = open(os.path.join(working_dir, glog_file_name), "r")
    pattern_date_time = re.compile(r'\d{4}\/\d{2}\/\d{2} \d{2}:\d{2}:\d{2}')
    start_time_str = pattern_date_time.findall(glog_log.readline())
    start_time = datetime.strptime(start_time_str[0], '%Y/%m/%d %H:%M:%S')

    for single_line in log.readlines():
        is_glog, level, time_raw = is_log(str(single_line))
        if not is_glog:
            continue

        if first_log:
            first_log = False
            last_hour = time_raw.hour

        # do day shift
        if time_raw.hour < last_hour:
            day_counter = day_counter + 1
        last_hour = time_raw.hour

        # the date of time_raw is 1900/1/1
        start_time_without_date = datetime(time_raw.year, time_raw.month, time_raw.day, hour=start_time.hour, minute=start_time.minute, second=start_time.second)
        time = time_raw - start_time_without_date + timedelta(days=day_counter)


        has_accuracy, accuracy = get_accuracy(str(single_line))
        if has_accuracy:
            df.loc[len(df.index)] = ["report_accuracy", time.total_seconds(), time_raw.strftime("%H:%M:%S"), start_time, single_node["name"], accuracy]
    df.to_csv(temp_file_path + single_node["name"] + ".csv")


if __name__ == "__main__":
    # find summary.json
    with open("summary.json", "rb") as file:
        data = file.read()
        summary = json.loads(data)

    # nodes
    if not os.path.exists(temp_file_path):
        os.mkdir(temp_file_path)
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        for single_node in summary["nodes"]:
            pool.apply_async(process_single_node, (single_node, ))
        pool.close()
        pool.join()

    # merge results
    dfs = []
    for single_entry in os.listdir(temp_file_path):
        if os.path.isdir(single_entry):
            continue
        temp_df = pandas.read_csv(temp_file_path + single_entry, header=0, index_col=0)
        dfs.append(temp_df)
    df = pandas.concat(dfs)
    print(df)

    # unify start time
    first_start_time_str = df.min(axis='rows')["start_time"]
    print(first_start_time_str)
    first_start_time = datetime.strptime(first_start_time_str, '%Y-%m-%d %H:%M:%S')
    def calculate_unify_time(row):
        start_time = datetime.strptime(row['start_time'], '%Y-%m-%d %H:%M:%S')
        unified_elapsed_time = row["elapsed"] + (start_time - first_start_time).total_seconds()
        return unified_elapsed_time

    df["unified_elapsed_time"] = df.apply(func=calculate_unify_time, axis='columns')
    df.sort_values("unified_elapsed_time", axis=0, inplace=True)

    # get line of each node
    df_for_each_node = {}
    node_names = df["node"].value_counts().index
    for node_name in node_names:
        df_for_each_node[node_name] = df.loc[(df.node == node_name) & (df.event_type == "report_accuracy")]

    # save df
    df.to_csv("./event_table.csv")

    fig = plt.figure()
    axs = fig.add_subplot(111)
    for node_name in node_names:
        df_for_this_node = df_for_each_node[node_name]
        axs.plot(df_for_this_node["unified_elapsed_time"], df_for_this_node["accuracy"], label=node_name, alpha=0.75)
    axs.grid()
    axs.set_title('accuracy')
    axs.set_xlabel('time (seconds)')
    axs.set_ylabel('accuracy (0-1)')
    axs.set_ylim([0, 1])
    axs.set_xlim([0, maximum_time_second])

    fig.tight_layout()
    fig.savefig('accuracy.pdf')
    fig.savefig('accuracy.jpg', dpi=800)
