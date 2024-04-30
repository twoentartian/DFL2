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

herd_effect_delay = calculate_herd_effect_delay(accuracy_df, weight_diff_df)

fig, axs = plt.subplots(2, figsize=(10, 10))

axs[0].axvline(x=herd_effect_delay, color='r', label=f'herd effect delay={herd_effect_delay}')
for col in accuracy_df.columns:
    if draw_only_first_node:
        if col == "0":
            axs[0].plot(accuracy_x, accuracy_df[col], label=col, alpha=0.75)
    else:
        axs[0].plot(accuracy_x, accuracy_df[col], label=col, alpha=0.75)

axs[1].axvline(x=herd_effect_delay, color='r', label=f'herd effect delay={herd_effect_delay}')
for col in weight_diff_df.columns:
    if numpy.sum(weight_diff_df[col]) == 0:
        continue
    axs[1].plot(weight_diff_x, weight_diff_df[col], label=col)

axs[0].grid()
axs[0].legend(ncol=5)
axs[0].set_title('accuracy')
axs[0].set_xlabel('time (tick)')
axs[0].set_ylabel('accuracy (0-1)')
axs[0].set_xlim([0, accuracy_df.index[accuracy_df_len-1]])
axs[0].set_ylim([0, 1])
if len(accuracy_df.columns) > 10:
    axs[0].legend().remove()

axs[1].grid()
axs[1].legend()
axs[1].set_title('model weight diff')
axs[1].set_xlabel('time (tick)')
axs[1].set_ylabel('weight diff')
axs[1].set_yscale('log')
axs[1].set_xlim([0, weight_diff_df.index[weight_diff_df_len-1]])

plt.tight_layout()
plt.savefig('accuracy_weight_diff_combine.pdf')
plt.savefig('accuracy_weight_diff_combine.jpg', dpi=800)
