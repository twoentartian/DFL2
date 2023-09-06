import math

import pandas as pd
import argparse
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

def calculate_herd_effect(size: int, start_degree: int, gamma: float, slope: float) -> float:
    df = pd.DataFrame()
    df["degree"] = range(start_degree, size)
    df["prob"] = df["degree"].apply(lambda x: pow(x, -gamma))
    prob_sum = df["prob"].sum()
    df["unified_prob"] = df["prob"].apply(lambda x: x/prob_sum)
    assert(abs(df["unified_prob"].sum() - 1) < 0.00001)
    df["accumulated_prob"] = df["unified_prob"].cumsum()
    df["having_at_least_one_node_above_degree"] = 1 - df["accumulated_prob"] ** size
    df["prob_largest_hub"] = df['having_at_least_one_node_above_degree'] - df['having_at_least_one_node_above_degree'].shift(-1)
    herd_effect_delay = df['prob_largest_hub'].multiply( (df['degree']/size) ** slope ).sum()

    return herd_effect_delay


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="match the herd effect in theory and practise", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("gamma", type=float, help="gamma of the scale free network")
    parser.add_argument("--start_degree", type=int, help="start degree of the scale free network", default=3)
    parser.add_argument("slope", type=float, help="the slope of herd effect delay in CM(k=3)+star")

    args = parser.parse_args()
    config = vars(args)
    if len(sys.argv) < 2:
        print("use: python ./herd_effect_delay_match_practical_value_theoretical_value.py {gamma} {slope}")
        exit(1)

    gamma = config["gamma"]
    start_degree = config["start_degree"]
    slope = config["slope"]

    # read files
    file_path = "./herd_effect_delay.csv"
    df_simulation = pd.read_csv(file_path)

    # calculate the herd effect in theory
    largest_network_size = df_simulation["size"].iloc[-1]
    print(f"largest_network_size:{largest_network_size}")
    digit_of_largest_network_size = len([int(digit) for digit in str(largest_network_size)])
    network_size = []
    for i in range(2, digit_of_largest_network_size):
        network_size = network_size + list(range(10**(i-1),10**i,2*10**(i-2)))
    network_size = network_size + list(range(10**(digit_of_largest_network_size-1),largest_network_size+1,2*10**(digit_of_largest_network_size-2)))

    df_theory = pd.DataFrame({'network_size': network_size})
    df_theory['herd_effect_delay'] = df_theory['network_size'].apply(lambda size: calculate_herd_effect(size, start_degree, gamma, slope))

    start_network_size = int(df_simulation['size'].iloc[0])
    start_theory_hed = float(df_theory['herd_effect_delay'].iloc[df_theory[df_theory['network_size'] == start_network_size].index].iloc[0])
    start_simulation_hed = float(df_simulation['herd_effect_delay'].iloc[df_simulation[df_simulation['size'] == start_network_size].index].iloc[0])
    print(f"start point: real={{ {start_network_size}:{start_simulation_hed} }} theory={{ {start_network_size}:{start_theory_hed} }}")

    end_network_size = int(df_simulation['size'].iloc[-1])
    end_theory_hed = float(df_theory['herd_effect_delay'].iloc[df_theory[df_theory['network_size'] == end_network_size].index].iloc[0])
    end_simulation_hed = float(df_simulation['herd_effect_delay'].iloc[df_simulation[df_simulation['size'] == end_network_size].index].iloc[0])
    print(f"end point: real={{ {end_network_size}:{end_simulation_hed} }} theory={{ {end_network_size}:{end_theory_hed} }}")

    scale_factor = (end_simulation_hed - start_simulation_hed) / (end_theory_hed - start_theory_hed)
    shift_factor = (start_simulation_hed - start_theory_hed * scale_factor) *0.5 + (end_simulation_hed - end_theory_hed * scale_factor) *0.5
    print(f"scale_factor: {scale_factor}     shift_factor: {shift_factor}")

    scale_factor = scale_factor
    shift_factor = shift_factor + 220
    print(f"after modification: scale_factor: {scale_factor}     shift_factor: {shift_factor}")

    df_theory['herd_effect_delay_map_to_simulation'] = df_theory['herd_effect_delay'] * scale_factor + shift_factor

    fig, axs = plt.subplots(1, 1, figsize=(10, 10), squeeze=False)
    axs[0, 0].plot(df_theory['network_size'], df_theory['herd_effect_delay_map_to_simulation'], label='theoretical')
    axs[0, 0].plot(df_simulation['size'], df_simulation['herd_effect_delay'], label='simulation')
    axs[0, 0].legend()
    axs[0, 0].grid()
    axs[0, 0].set_xscale('log')
    fig.savefig("comparision.pdf")
