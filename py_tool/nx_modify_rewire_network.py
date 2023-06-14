import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="rewire the network edges", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input_file", help="the topology file path", type=str)
    parser.add_argument("rewire_ratio", help="the ratio of rewiring edges", type=float)

    args = parser.parse_args()
    config = vars(args)



