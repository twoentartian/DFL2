import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

fusion_accuracy_mapping_file_path = "fusion_model_accuracy.csv"


def get_propagating_src(fusion_accuracy_mapping):
    propagating_src = [col for col in fusion_accuracy_mapping.columns if 'accuracy' not in col and 'loss' not in col]
    return propagating_src


def draw_fusion_accuracy_loss_graph(fusion_df, save_name):
    propagating_src = get_propagating_src(fusion_df)

    value_to_plot = "accuracy"
    if len(propagating_src) == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(fusion_df[propagating_src[0]], fusion_df[propagating_src[1]], fusion_df[propagating_src[2]], s=5, c=fusion_df[value_to_plot], cmap='hot')
        plt.colorbar(sc)
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_zlabel('Dimension 3')
        ax.set_title(f'3D Scatter Plot with {value_to_plot} as Temperature')
        plt.show()
    elif len(propagating_src) == 2:
        plt.figure()
        sc = plt.scatter(fusion_df[propagating_src[0]], fusion_df[propagating_src[1]], c=fusion_df[value_to_plot], s=5, cmap='viridis')
        plt.colorbar(sc, label='Accuracy')
        plt.title(f'Heatmap of {value_to_plot.capitalize()}')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 0')
        plt.savefig(save_name)
    elif len(propagating_src) == 1:
        plt.figure()
        plt.plot(fusion_df[propagating_src[0]], fusion_df[value_to_plot], marker='o', linestyle='-', color='royalblue')
        plt.title(f'Line Graph of {value_to_plot.capitalize()} Over Dimension "0"')
        plt.xlabel('Dimension "0"')
        plt.ylabel(value_to_plot.capitalize())
        plt.grid(True)
        plt.savefig(save_name)
    else:
        print(f'Invalid propagating source count {len(propagating_src)}')
        exit(-1)


if __name__ == "__main__":
    fusion_df = pd.read_csv(fusion_accuracy_mapping_file_path)
    draw_fusion_accuracy_loss_graph(fusion_df, "fusion_model_accuracy.pdf")