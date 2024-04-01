import sys

import numpy as np
import pandas as pd
from scipy import stats

if __name__ == "__main__":
    file_path = "herd_effect_delay.csv"
    df = pd.read_csv(file_path)

    all_options = ["size", "max_degree"]
    available_options = []
    for i in all_options:
        if i in df.columns:
            available_options.append(i)

    selected_index = 0
    if len(available_options) != 1:
        for index in range(len(available_options)):
            print(f"{index} --- {available_options[index]}")
        upper_bound = len(available_options)-1
        sys.stdout.write(f"please select the column as x axis[{0}~{upper_bound}]:")
        index = int(input())
        if index < 0 or index > upper_bound:
            print(f"invalid column index: {index}")
            exit(-1)
        selected_index = index
    else:
        selected_index = 0

    # sort column
    df["x(log)"] = np.log(df[available_options[selected_index]])
    df["y(log)"] = np.log(df["herd_effect_delay"])

    x = df["x(log)"].values
    y = df["y(log)"].values

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    # Calculating the 95% confidence interval for the slope (k)
    # The t-distribution critical value for 95% confidence and df = n - 2
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html
    t_crit = stats.t.ppf(1 - 0.025, len(x) - 2)

    # Confidence interval calculation
    k_conf_interval = (slope - t_crit * std_err, slope + t_crit * std_err)

    print(f"slope={slope}")
    print(f"intercept={intercept}")
    print(f"k_conf_interval={k_conf_interval}")
    print(f"std_err={std_err}")

    f = open("./slope.txt", "w")
    f.write(f"x axis is column:{available_options[selected_index]}\nslope:{slope}\nintercept:{intercept}\nk_conf_interval={k_conf_interval}\nstd_err={std_err}\n")
    f.close()