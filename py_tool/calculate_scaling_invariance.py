import math

import pandas as pd
import argparse
import sys
import matplotlib.pyplot as plt


def calculate_herd_effect(size: int, start_degree: int, gamma: float) -> float:
    df = pd.DataFrame()
    df["degree"] = range(start_degree, size)
    df["prob"] = df["degree"].apply(lambda x: pow(x, -gamma))
    prob_sum = df["prob"].sum()
    df["unified_prob"] = df["prob"].apply(lambda x: x/prob_sum)
    assert(abs(df["unified_prob"].sum() - 1) < 0.00001)
    df["accumulated_prob"] = df["unified_prob"].cumsum()
    df["having_at_least_one_node_above_degree"] = 1 - df["accumulated_prob"] ** size
    df["prob_largest_hub"] = df['having_at_least_one_node_above_degree'] - df['having_at_least_one_node_above_degree'].shift(-1)
    herd_effect_delay = df['prob_largest_hub'].multiply( (df['degree']/size) ** -0.19 ).sum()

    return herd_effect_delay


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="calculate the scaling invariance figure", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("gamma", type=float, help="gamma of the scale free network")
    parser.add_argument("--start_degree", type=int, help="start degree of the scale free network", default=3)
    parser.add_argument("largest_network_size", type=int, help="maximum network size")

    args = parser.parse_args()
    config = vars(args)
    if len(sys.argv) < 2:
        print("use: python ./calculate_scaling_invariance.py {network_size0} {network_size1} ...")
        exit(1)

    gamma = config["gamma"]
    start_degree = config["start_degree"]
    largest_network_size = config['largest_network_size']

    df = pd.DataFrame({'network_size': range(10, largest_network_size, math.floor((largest_network_size-10)/1000)) })
    df['herd_effect_delay'] = df['network_size'].apply(lambda size: calculate_herd_effect(size, start_degree, gamma))
    df.to_csv('scaling_invariance.csv')

    fig, axs = plt.subplots(2, 1, figsize=(10, 10), squeeze=False)
    axs[0, 0].plot(df['network_size'], df['herd_effect_delay'])
    axs[0, 0].set_xlabel('network size')
    axs[0, 0].set_ylabel('estimated herd effect delay')

    axs[1, 0].plot(df['network_size'], df['herd_effect_delay'])
    axs[1, 0].set_xlabel('network size')
    axs[1, 0].set_ylabel('estimated herd effect delay')
    axs[1, 0].set_xscale('log')
    fig.savefig('scaling_invariance.pdf')
