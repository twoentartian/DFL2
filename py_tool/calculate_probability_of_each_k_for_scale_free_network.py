import pandas as pd
import argparse
import sys

if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser(description="calculate the probability of each degree for a scale free network", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("gamma", type=float, help="gamma of the scale free network")
    parser.add_argument("--start_degree", type=int, help="start degree of the scale free network", default=3)
    parser.add_argument("network_sizes", nargs='+', help="network size")

    args = parser.parse_args()
    config = vars(args)
    if len(sys.argv) < 2:
        print("use: python ./calculate_probability_of_each_k_for_scale_free_network.py {network_size0} {network_size1} ...")
        exit(1)

    gamma = config["gamma"]
    start_degree = config["start_degree"]

    for size_str in config["network_sizes"]:
        size = int(size_str)
        df = pd.DataFrame()
        df["degree"] = range(start_degree, size)
        df["prob"] = df["degree"].apply(lambda x: pow(x, -gamma))
        prob_sum = df["prob"].sum()
        df["unified_prob"] = df["prob"].apply(lambda x: x/prob_sum)
        assert(abs(df["unified_prob"].sum() - 1) < 0.00001)

        print(df)