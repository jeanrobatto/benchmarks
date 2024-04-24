import os
import matplotlib.pyplot as plt


class TrainTimeGraph:
    """
    This class provides functionality to plot and save training time over epochs.

    It tracks the time taken for each epoch during training and provides a method to visualize
    the time taken per epoch and the cumulative time over epochs in a single graph.
    """

    def __init__(self):
        """
        Initialize an instance of TrainTimeGraph with an empty dictionary to store epoch-wise times.
        """
        self.epoch_train_time = {}

    def update_epoch_train_time(self, epoch, train_time):
        """
        Update the dictionary of epoch-wise training times.

        Args:
            epoch (int): The epoch number.
            train_time (float): The train time for the epoch in seconds.

        Returns:
            None
        """
        self.epoch_train_time[epoch] = train_time

    def print_graph(self, destination='graphs/', filename='train_time_plot.pdf'):
        """
        Plot and save the train time graph to a specified destination directory.

        Args:
            destination (str, optional): The directory where the graph image will be saved, defaults to 'graphs/'.
            filename (str, optional): The filename of the saved graph image, defaults to 'train_time_plot.pdf'.

        Returns:
            None

        Example:
            >>> train_time_graph = TrainTimeGraph()
            >>> train_time_graph.print_graph(destination='output/', filename='train_time_plot.png')
        """

        # Extracting x (epochs) and y (train time) data from the dictionary
        epochs = list(self.epoch_train_time.keys())
        train_time = list(self.epoch_train_time.values())
        cumulative_train_time = [sum(train_time[:i + 1]) for i in range(len(train_time))]

        # Plotting the lines
        plt.plot(epochs, train_time, marker='o', markersize=0.2, linestyle='-', linewidth=0.8, label='Train Time per Epoch')
        plt.plot(epochs, cumulative_train_time, marker='o', markersize=0.2, linestyle='-', linewidth=0.8, label='Cumulative Train Time')

        # Adding labels and title
        plt.xlabel('Epochs')
        plt.ylabel('Train Time')
        plt.title('Train Time (seconds) over Epochs')

        # Displaying the plot
        plt.grid(True)

        # Add legend
        plt.legend()

        # Save to destination path
        # Create the directory if it doesn't exist
        if not os.path.exists(destination):
            os.makedirs(destination)

        path = os.path.join(destination, filename)
        plt.savefig(path)

        plt.close()
        plt.clf()
