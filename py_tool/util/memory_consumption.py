from matplotlib import pyplot as pt
import pandas
import re

target = "DFL_simulator_opti"

if __name__ == '__main__':
    data_file = open(target + ".INFO", "r")
    data_lines = data_file.readlines()

    memory_df = pandas.DataFrame(columns=["memory:MB"])

    current_tick = 0

    for line in data_lines:
        tick = re.findall(r"tick: \d+", line)
        if len(tick) != 0:
            tick = tick[0][6:]
            current_tick = int(tick)
        memory_consumption = re.findall(r"\d+ MB", line)
        if len(memory_consumption) != 0:
            memory_consumption = memory_consumption[0][0:-4]
            memory_consumption = int(memory_consumption)
            memory_df.loc[current_tick] = {"memory:MB": memory_consumption}

    memory_df.to_csv("memory_consumption_" + target + ".csv")

    fig, ax = pt.subplots(1, 1)
    x = memory_df.index
    y = memory_df["memory:MB"]
    ax.plot(x, y)
    ax.grid(which='both')
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)
    fig.savefig("memory_consumption_" + target + ".jpg")
    fig.savefig("memory_consumption_" + target + ".pdf")