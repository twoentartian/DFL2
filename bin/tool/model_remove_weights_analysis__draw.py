import matplotlib.pyplot as plt
import pandas as pd


if __name__ == "__main__":
    # Load the uploaded CSV file to check its contents
    file_path = '/mnt/data/accuracy_of_compressed_models.csv'
    data = pd.read_csv(file_path)

    # Display the first few rows of the dataframe to understand its structure
    data.head()

    # Scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(data['compress ratio'], data['accuracy'], alpha=0.6)

    # Title and labels
    plt.title('Scatter Plot of Compression Ratio vs. Accuracy')
    plt.xlabel('Compression Ratio')
    plt.ylabel('Accuracy')

    # Show the plot
    plt.grid(True)
    plt.savefig("model_remove_weights_analysis.pdf")
