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


def moving_average(data, window_size):
    return data.rolling(window=window_size, min_periods=1).mean()


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
    smallest_network_size = df_simulation["size"].iloc[0]
    largest_network_size = df_simulation["size"].iloc[-1]
    print(f"largest_network_size:{largest_network_size}")
    digit_of_smallest_network_size = len([int(digit) for digit in str(smallest_network_size)])
    digit_of_largest_network_size = len([int(digit) for digit in str(largest_network_size)])
    network_size = []
    network_size = network_size + list(range(smallest_network_size,10**digit_of_smallest_network_size,2*10**(digit_of_smallest_network_size-2)))
    for i in range(digit_of_smallest_network_size+1, digit_of_largest_network_size):
        network_size = network_size + list(range(10**(i-1),10**i,2*10**(i-2)))
    network_size = network_size + list(range(10**(digit_of_largest_network_size-1),largest_network_size+1,2*10**(digit_of_largest_network_size-2)))
    network_size = network_size + list(df_simulation['size'].to_numpy(dtype=int))
    network_size = list(dict.fromkeys(network_size))
    network_size.sort()
    print(network_size)

    df_theory = pd.DataFrame({'network_size': network_size})
    df_theory['herd_effect_delay'] = df_theory['network_size'].apply(lambda size: calculate_herd_effect(size, start_degree, gamma, slope))

    theory_hed = []
    simulation_hed = []
    matched_network_sizes = []
    for matched_network_size in df_simulation['size'].to_numpy():
        theory_hed_now = float(df_theory['herd_effect_delay'].iloc[df_theory[df_theory['network_size'] == matched_network_size].index].iloc[0])
        theory_hed.append(theory_hed_now)
        simulation_hed_now = float(df_simulation['herd_effect_delay'].iloc[df_simulation[df_simulation['size'] == matched_network_size].index].iloc[0])
        simulation_hed.append(simulation_hed_now)
        matched_network_size_now = int(matched_network_size)
        matched_network_sizes.append(matched_network_size_now)
        print(f"simulation={{ {matched_network_size_now}:{simulation_hed_now} }} theory={{ {matched_network_size_now}:{theory_hed_now} }}")

    theory_hed = np.array(theory_hed)
    simulation_hed = np.array(simulation_hed)
    matched_network_sizes = np.array(matched_network_sizes).reshape((-1, 1))

    model = LinearRegression()
    model.fit(matched_network_sizes, theory_hed)
    slope_theory = model.coef_[0]
    intercept_theory = model.intercept_

    model = LinearRegression()
    model.fit(matched_network_sizes, simulation_hed)
    slope_simulation = model.coef_[0]
    intercept_simulation = model.intercept_

    scale_factor = slope_simulation / slope_theory
    shift_factor = intercept_simulation - scale_factor * intercept_theory
    print(f"scale_factor: {scale_factor}     shift_factor: {shift_factor}")

    scale_factor = scale_factor
    shift_factor = shift_factor
    print(f"after manual modification: scale_factor: {scale_factor}     shift_factor: {shift_factor}")

    df_theory['herd_effect_delay_map_to_simulation'] = df_theory['herd_effect_delay'] * scale_factor + shift_factor

    fig, axs = plt.subplots(1, 1, figsize=(10, 10), squeeze=False)
    axs[0, 0].plot(df_theory['network_size'], df_theory['herd_effect_delay_map_to_simulation'], label='theoretical')
    axs[0, 0].plot(df_simulation['size'], df_simulation['herd_effect_delay'], label='simulation')
    smooth_window_size = 8
    df_simulation['smoothed_herd_effect_delay'] = moving_average(df_simulation['herd_effect_delay'], smooth_window_size)
    axs[0, 0].plot(df_simulation['size'], df_simulation['smoothed_herd_effect_delay'], label=f'simulation(smoothed-{smooth_window_size}')
    axs[0, 0].legend()
    axs[0, 0].grid()
    axs[0, 0].set_xscale('log')
    fig.savefig("comparision.pdf")
