import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

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

    x = df["x(log)"].values.reshape(-1, 1)
    y = df["y(log)"].values
    model = LinearRegression()
    model.fit(x, y)
    slope = model.coef_[0]
    intercept = model.intercept_
    print(f"slope:{slope}")
    print(f"intercept:{intercept}")

    f = open("./slope.txt", "w")
    f.write(f"x axis is column:{available_options[selected_index]}\nslope:{slope}\nintercept:{intercept}\n")
    f.close()