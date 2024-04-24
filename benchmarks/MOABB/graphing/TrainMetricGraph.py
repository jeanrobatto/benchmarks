import os
import matplotlib.pyplot as plt


class TrainMetricGraph:
    """
    This class provides functionality to plot and save train metrics over epochs.

    It tracks metrics such as train loss, validation F1 score, validation accuracy, and validation loss
    over multiple epochs and provides a method to visualize these metrics in a single graph.
    """

    def __init__(self):
        """
        Initialize an instance of TrainMetricGraph with empty dictionaries to store epoch-wise metrics.
        """
        self.epoch_train_loss = {}
        self.epoch_valid_f1 = {}
        self.epoch_valid_acc = {}
        self.epoch_valid_loss = {}

    def update_epoch_train_loss(self, epoch, train_loss):
        """
        Update the dictionary of epoch-wise train losses.

        Args:
            epoch (int): The epoch number.
            train_loss (float): The training loss for the epoch.

        Returns:
            None
        """
        self.epoch_train_loss[epoch] = train_loss

    def update_epoch_valid_f1(self, epoch, f1):
        """
        Update the dictionary of epoch-wise validation F1 scores.

        Args:
            epoch (int): The epoch number.
            f1 (float): The validation F1 score for the epoch.

        Returns:
            None
        """
        self.epoch_valid_f1[epoch] = f1

    def update_epoch_valid_acc(self, epoch, valid_acc):
        """
        Update the dictionary of epoch-wise validation accuracies.

        Args:
            epoch (int): The epoch number.
            valid_acc (float): The validation accuracy for the epoch.

        Returns:
            None
        """
        self.epoch_valid_acc[epoch] = valid_acc

    def update_epoch_valid_loss(self, epoch, valid_loss):
        """
        Update the dictionary of epoch-wise validation losses.

        Args:
            epoch (int): The epoch number.
            valid_loss (float): The validation loss for the epoch.

        Returns:
            None
        """
        self.epoch_valid_loss[epoch] = valid_loss

    def print_graph(self, destination='graphs/', filename='train_metrics_plot.pdf'):
        """
        Plot and save the train metrics graph to a specified destination directory.

        Args:
            destination (str, optional): The directory where the graph image will be saved, defaults to 'graphs/'.
            filename (str, optional): The filename of the saved graph image, defaults to 'train_metrics_plot.pdf'.

        Returns:
            None

        Example:
            >>> train_metrics = TrainMetricGraph()
            >>> train_metrics.update_epoch_train_loss(1, 0.5)
            >>> train_metrics.update_epoch_valid_f1(1, 0.7)
            >>> train_metrics.update_epoch_valid_acc(1, 0.8)
            >>> train_metrics.update_epoch_valid_loss(1, 0.3)
            >>> train_metrics.print_graph(destination='output/', filename='metrics_plot.png')
        """

        # Extracting x (epochs) and y (train loss) data from the dictionary
        epochs = list(self.epoch_train_loss.keys())
        train_loss = list(self.epoch_train_loss.values())
        valid_f1 = list(self.epoch_valid_f1.values())
        valid_acc = list(self.epoch_valid_acc.values())
        valid_loss = list(self.epoch_valid_loss.values())

        # Plotting the lines
        plt.plot(epochs, train_loss, marker='o', markersize=0.2, linestyle='-', linewidth=0.8, label='train loss')
        plt.plot(epochs, valid_f1, marker='o', markersize=0.2, linestyle='-', linewidth=0.8, label='valid f1')
        plt.plot(epochs, valid_acc, marker='o', markersize=0.2, linestyle='-', linewidth=0.8, label='valid acc')
        plt.plot(epochs, valid_loss, marker='o', markersize=0.2, linestyle='-', linewidth=0.8, label='valid loss')

        # Adding labels and title
        plt.xlabel('Epochs')
        plt.ylabel('Value')
        plt.title('Train Metrics over Epochs')

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
