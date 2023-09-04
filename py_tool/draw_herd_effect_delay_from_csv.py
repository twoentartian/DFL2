import pandas as pd
import matplotlib.pyplot as plt
import os

herd_effect_draw_with_size = True
x_size = 10
y_size = 5

if __name__ == "__main__":
    herd_effect_delay_df: pd.DataFrame = pd.read_csv('herd_effect_delay.csv')
    print(herd_effect_delay_df)

    output_path = "./herd_effect_delay_images"
    if not os.path.exists(output_path):
        os.mkdir("herd_effect_delay_images")

    if herd_effect_draw_with_size:
        # raw
        fig, axs = plt.subplots(1, 1, figsize=(x_size, y_size))
        axs.plot(herd_effect_delay_df["size"], herd_effect_delay_df['herd_effect_delay'])
        axs.set_xlabel('network name')
        axs.set_ylabel('herd effect delay (tick)')
        axs.grid()
        fig.savefig(os.path.join(output_path, 'raw.pdf'))
        fig.savefig(os.path.join(output_path, 'raw.jpg'))

        # x axis is log
        fig, axs = plt.subplots(1, 1, figsize=(x_size, y_size))
        axs.plot(herd_effect_delay_df["size"], herd_effect_delay_df['herd_effect_delay'])
        axs.set_xlabel('network size')
        axs.set_xscale('log')
        axs.set_ylabel('herd effect delay (tick)')
        axs.grid()
        fig.savefig(os.path.join(output_path, 'x_is_log.pdf'))
        fig.savefig(os.path.join(output_path, 'x_is_log.jpg'))

        # x axis is log, y axis is smoothed (15)
        fig, axs = plt.subplots(1, 1, figsize=(x_size, y_size))
        smooth_window_size = 15

        def moving_average(data, window_size):
            return data.rolling(window=window_size, min_periods=1).mean()
        herd_effect_delay_df['smoothed_herd_effect_delay'] = moving_average(herd_effect_delay_df['herd_effect_delay'], smooth_window_size)
        axs.plot(herd_effect_delay_df["size"], herd_effect_delay_df['smoothed_herd_effect_delay'])
        axs.set_xlabel('network size(log)')
        axs.set_xscale('log')
        axs.set_ylabel('herd effect delay (tick)(smoothed ' + str(smooth_window_size) + ')')
        axs.grid()
        fig.savefig(os.path.join(output_path, '15_smoothed.pdf'))
        fig.savefig(os.path.join(output_path, '15_smoothed.jpg'))

    else:
        # raw
        fig, axs = plt.subplots(1, 1, figsize=(x_size, y_size))
        axs.plot(herd_effect_delay_df["network_name"], herd_effect_delay_df['herd_effect_delay'])
        axs.set_xlabel('network name')
        axs.set_ylabel('herd effect delay (tick)')
        axs.grid()
        fig.savefig(os.path.join(output_path, 'raw.pdf'))
        fig.savefig(os.path.join(output_path, 'raw.jpg'))


