import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

herd_effect_draw_with_size = True
x_size = 10
y_size = 5

if __name__ == "__main__":
    herd_effect_delay_df: pd.DataFrame = pd.read_csv('herd_effect_delay.csv')

    all_options = ["size", "max_degree"]
    available_options = []
    for i in all_options:
        if i in herd_effect_delay_df.columns:
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

    sorted_herd_effect_delay_df = herd_effect_delay_df.sort_values(by=[available_options[selected_index]])
    print(sorted_herd_effect_delay_df)

    output_path = "."
    if not os.path.exists(output_path):
        os.mkdir("herd_effect_delay_images")

    # x-axis and y-axis are log
    fig, axs = plt.subplots(2, 1, figsize=(x_size, y_size))
    axs[0].plot(sorted_herd_effect_delay_df[available_options[selected_index]], sorted_herd_effect_delay_df['herd_effect_delay'])
    axs[0].set_xlabel(available_options[selected_index])
    axs[0].set_ylabel('herd effect delay (tick)')
    axs[0].grid()
    axs[1].plot(sorted_herd_effect_delay_df[available_options[selected_index]], sorted_herd_effect_delay_df['herd_effect_delay'])
    axs[1].set_xlabel(f"{available_options[selected_index]}")
    axs[1].set_ylabel('herd effect delay (tick)')
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')
    axs[1].grid()
    fig.savefig(os.path.join(output_path, 'herd_effect_delay_to_column.pdf'))
    fig.savefig(os.path.join(output_path, 'herd_effect_delay_to_column.jpg'))

