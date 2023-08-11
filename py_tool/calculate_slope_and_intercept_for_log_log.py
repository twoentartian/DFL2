import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":
    file_path = "herd_effect_delay.csv"
    df = pd.read_csv(file_path)
    df["size(log)"] = np.log(df["size"])
    df["herd_effect_delay(log)"] = np.log(df["herd_effect_delay"])

    x = df["size(log)"].values.reshape(-1, 1)
    y = df["herd_effect_delay(log)"].values

    model = LinearRegression()
    model.fit(x, y)

    slope = model.coef_[0]
    intercept = model.intercept_

    print(f"slope:{slope}")
    print(f"intercept:{intercept}")