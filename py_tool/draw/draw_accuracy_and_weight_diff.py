import os.path

import pandas
import matplotlib.pyplot as plt
import numpy

draw_only_first_node = False
enable_draw_every_tick = False
draw_every_tick = 500


def calculate_herd_effect_delay(arg_accuracy_df: pandas.DataFrame, arg_model_weight_diff_df: pandas.DataFrame):
    average_accuracy: pandas.Series = arg_accuracy_df.mean(axis=1)
    average_accuracy_diff = average_accuracy.diff()
    average_accuracy_diff.dropna(inplace=True)
    largest_diff = average_accuracy_diff.nlargest(10)
    largest_indexes = largest_diff.index
    for i in largest_indexes:
        if i > 20:
            return i


accuracy_file_path = 'accuracy.csv'
accuracy_df = pandas.read_csv(accuracy_file_path, index_col=0, header=0)
if enable_draw_every_tick:
    accuracy_df = accuracy_df[accuracy_df.index % draw_every_tick ==0]
print(accuracy_df)
accuracy_x = accuracy_df.index
accuracy_df_len = len(accuracy_df)

weight_diff_file_path = 'model_weight_diff.csv'
weight_diff_df = pandas.read_csv(weight_diff_file_path, index_col=0, header=0)
print(weight_diff_df)
weight_diff_x = weight_diff_df.index
weight_diff_df_len = len(weight_diff_df)

loss_file_path = 'loss.csv'
loss_df = None
if os.path.exists(loss_file_path):
    loss_df = pandas.read_csv(loss_file_path, index_col=0, header=0)
    print(loss_df)

herd_effect_delay = calculate_herd_effect_delay(accuracy_df, weight_diff_df)

num_of_plots = 2
if loss_df is not None:
    num_of_plots = num_of_plots + 1

fig, axs = plt.subplots(num_of_plots, figsize=(14, 7*num_of_plots))

plot_index = 0

###################### accuracy
axs[plot_index].axvline(x=herd_effect_delay, color='r', label=f'herd effect delay={herd_effect_delay}')
for col in accuracy_df.columns:
    if draw_only_first_node:
        if col == "0":
            axs[plot_index].plot(accuracy_x, accuracy_df[col], label=col, alpha=0.75)
    else:
        axs[plot_index].plot(accuracy_x, accuracy_df[col], label=col, alpha=0.75)

axs[plot_index].grid()
axs[plot_index].legend(ncol=5)
axs[plot_index].set_title('accuracy')
axs[plot_index].set_xlabel('time (tick)')
axs[plot_index].set_ylabel('accuracy (0-1)')
axs[plot_index].set_xlim([0, accuracy_df.index[accuracy_df_len-1]])
axs[plot_index].set_ylim([0, 1])
if len(accuracy_df.columns) > 10:
    axs[plot_index].legend().remove()
plot_index = plot_index + 1

###################### loss
if loss_df is not None:
    loss_x = loss_df.index
    loss_df_len = len(loss_df)
    for col in loss_df.columns:
        axs[plot_index].plot(loss_x, loss_df[col], label=col, alpha=0.75)
    axs[plot_index].grid()
    axs[plot_index].legend(ncol=5)
    axs[plot_index].set_title('loss')
    axs[plot_index].set_xlabel('time (tick)')
    axs[plot_index].set_ylabel('loss')
    axs[plot_index].set_xlim([0, loss_df.index[loss_df_len-1]])
    if len(loss_df.columns) > 10:
        axs[plot_index].legend().remove()
    plot_index = plot_index + 1

###################### weight diff
axs[plot_index].axvline(x=herd_effect_delay, color='r', label=f'herd effect delay={herd_effect_delay}')
for col in weight_diff_df.columns:
    if numpy.sum(weight_diff_df[col]) == 0:
        continue
    axs[plot_index].plot(weight_diff_x, weight_diff_df[col], label=col)

axs[plot_index].grid()
axs[plot_index].legend()
axs[plot_index].set_title('model weight diff')
axs[plot_index].set_xlabel('time (tick)')
axs[plot_index].set_ylabel('weight diff')
axs[plot_index].set_yscale('log')
axs[plot_index].set_xlim([0, weight_diff_df.index[weight_diff_df_len-1]])
if len(weight_diff_df.columns) > 10:
    axs[plot_index].legend().remove()
plot_index = plot_index + 1

plt.tight_layout()
plt.savefig('accuracy_weight_diff_combine.pdf')
plt.savefig('accuracy_weight_diff_combine.jpg', dpi=800)
