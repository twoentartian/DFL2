import math

import pandas as pd
import argparse
import sys
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text


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
    parser = argparse.ArgumentParser(description="calculate scaling curves of different slopes", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("gamma", type=float, help="gamma of the scale free network")
    parser.add_argument("--start_degree", type=int, help="start degree of the scale free network", default=3)
    parser.add_argument("largest_network_size", type=int, help="maximum network size")

    args = parser.parse_args()
    config = vars(args)
    if len(sys.argv) < 2:
        print("use: python ./calculate_scaling_curves_of_different_slopes.py {gamma} {largest_network_size} ...")
        exit(1)

    gamma = config["gamma"]
    start_degree = config["start_degree"]
    largest_network_size = config['largest_network_size']
    slopes = np.linspace(start=-2, stop=-0.001, num=6)

    fig, axs = plt.subplots(3, 1, figsize=(12, 8), squeeze=False)
    fig_out, axs_out = plt.subplots(1, 1, figsize=(6, 3), squeeze=True)

    annotations0 = []
    annotations1 = []
    annotations2 = []
    annotations_out = []
    for slope_index in range(len(slopes)):
        slope = slopes[slope_index]
        print(f"calculating slope={slope}")

        df = pd.DataFrame({'network_size': range(10, largest_network_size, math.floor((largest_network_size-10)/1000)) })
        df['herd_effect_delay'] = df['network_size'].apply(lambda size: calculate_herd_effect(size, start_degree, gamma, slope))

        last_df_x = float(df['network_size'].tail(1).iloc[0])
        last_df_y = float(df['herd_effect_delay'].tail(1).iloc[0])

        axs[0, 0].plot(df['network_size'], df['herd_effect_delay'], label="k={:.3f}".format(slope))
        annotations0.append(axs[0, 0].annotate("{:.3f}".format(slope), (last_df_x, last_df_y)))
        axs[0, 0].set_xlabel('network size')
        axs[0, 0].set_ylabel('estimated herd effect delay')

        axs[1, 0].plot(df['network_size'], df['herd_effect_delay'], label="k={:.3f}".format(slope))
        annotations1.append(axs[1, 0].annotate("{:.3f}".format(slope), (last_df_x, last_df_y)))
        axs[1, 0].set_xlabel('network size')
        axs[1, 0].set_ylabel('estimated herd effect delay (log)')
        axs[1, 0].set_xscale('log')

        axs[2, 0].plot(df['network_size'], df['herd_effect_delay'], label="k={:.3f}".format(slope))
        annotations2.append(axs[2, 0].annotate("{:.3f}".format(slope), (last_df_x, last_df_y)))
        axs[2, 0].set_xlabel('network size')
        axs[2, 0].set_ylabel('estimated herd effect delay (log)')
        axs[2, 0].set_xscale('log')
        axs[2, 0].set_yscale('log')

        axs_out.plot(df['network_size'], df['herd_effect_delay'], label="m={:.1f}".format(slope))
        if slope_index == len(slopes)-1 or slope_index == 0:
            annotations_out.append(axs_out.annotate("m={:.1f}".format(slope), (last_df_x, last_df_y)))
        axs_out.set_xlabel('network size (log)')
        axs_out.set_ylabel('estimated herd effect delay (log)')
        axs_out.set_xscale('log')

    # adjust_text(annotations0)
    # adjust_text(annotations1)
    adjust_text(annotations_out)
    axs[0, 0].legend()
    axs[0, 0].grid(which='major', color='#DDDDDD', linewidth=0.8)
    axs[0, 0].grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
    axs[1, 0].legend()
    axs[1, 0].grid(which='major', color='#DDDDDD', linewidth=0.8)
    axs[1, 0].grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
    axs[2, 0].legend()
    axs[2, 0].grid(which='major', color='#DDDDDD', linewidth=0.8)
    axs[2, 0].grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
    fig.tight_layout()
    fig.savefig('scaling_invariance_of_different_slopes.pdf',bbox_inches='tight',pad_inches=0.05)

    # adjust_text(annotations_out)
    axs_out.legend()
    axs_out.grid(which='major', color='#DDDDDD', linewidth=0.8)
    axs_out.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
    fig_out.tight_layout()
    fig_out.savefig('scaling_invariance_of_different_slopes_out.pdf',bbox_inches='tight',pad_inches=0.05)