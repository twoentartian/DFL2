import pandas
import matplotlib
import re
import json
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

log_file_name = "./node_log.txt"
glog_file_name = "./log/DFL.INFO"


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
    pattern_accuracy = re.compile(r'accuracy: \d.\d{2}')

    # accuracy
    accuracy_str = pattern_accuracy.findall(msg)
    if not accuracy_str:
        return False, None
    accuracy_str = accuracy_str[0]
    accuracy_str = accuracy_str[10:]
    return True, float(accuracy_str)


if __name__ == "__main__":
    # find summary.json
    with open("summary.json", "rb") as file:
        data = file.read()
        summary = json.loads(data)

    df = pandas.DataFrame(columns=["event_type", "elapsed", "time", "start_time", "node", "accuracy"])
    # nodes
    for single_node in summary["nodes"]:
        print("checking node: " + single_node["name"])
        working_dir = os.path.join(os.curdir, single_node["folder"])
        log = open(os.path.join(working_dir, log_file_name), "r")

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

    # unify start time
    first_start_time = df.min(axis='rows')["start_time"]


    def calculate_unify_time(row):
        unified_elapsed_time = row["elapsed"] + (row['start_time'] - first_start_time).total_seconds()
        return unified_elapsed_time


    df["unified_elapsed_time"] = df.apply(func=calculate_unify_time, axis='columns')
    df.sort_values("unified_elapsed_time", axis=0, inplace=True)
    print(df)

    # get line of each node
    df_for_each_node = {}
    node_names = df["node"].value_counts().index
    for node_name in node_names:
        df_for_each_node[node_name] = df.loc[(df.node == node_name) & (df.event_type == "report_accuracy")]

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

    fig.tight_layout()
    fig.savefig('accuracy.pdf')
    fig.savefig('accuracy.jpg', dpi=800)
