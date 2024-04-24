import matplotlib.pyplot as plt
import os


class EegDataGraph:
    """
    This class provides functionality to plot and save EEG data in a graph.

    It takes EEG data as input, which is expected to be in the shape of (time_points, num_channels, 1),
    and generates a separate line plot for each channel, with time on the x-axis and amplitude on the y-axis.

    """

    def print_graph(self, eeg_data, destination='graphs/',
                    filename='eeg_plot_example.pdf'):
        """
        Plot and save the EEG data graph to a specified destination directory.

        Args:
            eeg_data (numpy.ndarray): The EEG data to be plotted, expected shape (time_points, num_channels, 1).
            destination (str, optional): The directory where the graph image will be saved, defaults to 'graphs/'.
            filename (str, optional): The filename of the saved graph image, defaults to 'eeg_plot_example.pdf'.

        Returns:
            None

        Example:
        >>> eeg_graph = EegDataGraph()
        >>> eeg_data = np.random.randn(500, 3, 1)  # Example EEG data with 500 time points and 3 channels
        >>> eeg_graph.print_graph(eeg_data, destination='output/', filename='eeg_plot.png')
        """

        # Extract the number of channels and time points
        time_points = eeg_data.shape[0]
        num_channels = eeg_data.shape[1]

        # Create a figure and axes object
        _, axes = plt.subplots(num_channels, 1, figsize=(15, 6 * num_channels))

        # Plot each channel's data
        for i in range(num_channels):
            # Extract data for the current channel
            channel_data = eeg_data[:, i, 0].cpu().detach().numpy()

            # Plot the channel's data as a line graph
            axes[i].plot(range(time_points), channel_data)

            # Add labels and title for each subplot
            axes[i].set_title(f'EEG Channel {i+1}')
            axes[i].set_xlabel('Time (seconds)')
            axes[i].set_ylabel('Amplitude')

        plt.subplots_adjust(top=0.95, bottom=0.05, hspace=0.4)

        # Save to destination path
        # Create the directory if it doesn't exist
        if not os.path.exists(destination):
            os.makedirs(destination)

        path = os.path.join(destination, filename)
        plt.savefig(path)

        plt.close()
        plt.clf()
