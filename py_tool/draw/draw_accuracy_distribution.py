import math
import os
import sys

import numpy
import numpy as np
import pandas
import matplotlib.pyplot as plt
from matplotlib import colors

accuracy_file_path = "./accuracy.csv"
n_bins = 20

if __name__ == "__main__":
    accuracy_df = pandas.read_csv(accuracy_file_path, index_col=0, header=0)

    number_of_figure = len(accuracy_df. index)
    plot_row = plot_col = math.ceil(math.sqrt(number_of_figure))

    fig, axs = plt.subplots(plot_row, plot_col, tight_layout=True, squeeze=False, figsize=[plot_row*4, plot_col*4])

    #find the maximum height
    maximum_height = 0
    for index, row in accuracy_df.iterrows():
        counts = row.value_counts()
        current_maximum_height = counts.max()
        if current_maximum_height > maximum_height:
            maximum_height = current_maximum_height

    figure_count = 0
    for index, row in accuracy_df.iterrows():
        print("processing tick " + str(index))

        figure_x = figure_count // plot_row
        figure_y = figure_count % plot_row
        current_axs = axs[figure_x, figure_y]
        figure_count = figure_count + 1

        # N is the count in each bin, bins is the lower-limit of the bin
        N, bins, patches = current_axs.hist(row, bins=n_bins)

        # We'll color code by height, but you could use any scalar
        fracs = N / N.max()

        # we need to normalize the data to 0..1 for the full range of the colormap
        norm = colors.Normalize(fracs.min(), fracs.max())

        # Now, we'll loop through our objects and set the color of each accordingly
        for thisfrac, thispatch in zip(fracs, patches):
            color = plt.cm.viridis(norm(thisfrac))
            thispatch.set_facecolor(color)

        current_axs.set_title('Tick: ' + str(index) + " accuracy distribution")
        current_axs.set_xlim([0, 1])
        current_axs.set_ylim([0, maximum_height])
        current_axs.set_xlabel('accuracy')
        current_axs.set_ylabel('distribution')

    print("saving figures")
    fig.savefig("accuracy_distribution.jpg")
    fig.savefig("accuracy_distribution.pdf")
