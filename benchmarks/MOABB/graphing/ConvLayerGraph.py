import os

import matplotlib.pyplot as plt
import numpy as np
import torch


class ConvLayerGraph:
    """
        This class draws and saves to disk a custom graph designed
        to show the intermediate feature representation of some EEG input
        following a number of convolution blocks.
    """

    def print_graph(self, data, destination='graphs/',
                    filename='eeg_conv_plot_example.pdf', show_attention=False, attention_map=None):
        """
        Print and save to disk a custom graph showing the intermediate feature representation of EEG input data
        after convolution blocks.

        Args:
            data (torch.Tensor): The EEG input data after convolution block operations. 
                                 Shape: (C, h, T)
            destination (str, optional): The directory where the graph image will be saved, defaults to 'graphs/'.
            filename (str, optional): The filename of the saved graph image, defaults to 'eeg_conv_plot.pdf'.
            show_attention (bool, optional): Flag indicating whether to show attention map overlay, defaults to False.
            attention_map (torch.Tensor, optional): The attention map to be displayed, defaults to None.

        Returns:
            None

        Example:
            >>> conv_layer = ConvLayerGraph()
            >>> data = torch.randn(3, 1, 100)  # Example EEG input data
            >>> conv_layer.print_graph(data, destination='output/', filename='eeg_conv_plot.png', show_attention=True)
        """

        # Extracting the number of channels and time points
        time_points = data.shape[2]
        num_channels = data.shape[0]

        # Create a figure and axes object
        _, axes = plt.subplots(num_channels, 1, figsize=(15, 6 * num_channels))

        # Get attention map
        if show_attention:
            final_attention_map = torch.mean(attention_map, dim=0)

        # Plot each channel's data
        for i in range(num_channels):
            # Extract data for the current channel
            channel_data = data[i, 0, :].cpu().detach().numpy()

            # Plot the channel's data as a line graph
            axes[i].plot(range(time_points), channel_data)

            # Add labels and title for each subplot
            axes[i].set_title(f'EEG Channel {i+1}')
            axes[i].set_xlabel('Time (seconds)')
            axes[i].set_ylabel('Value')

            if show_attention:

                filename = 'eeg_conv_plot_with_attention_example.pdf'

                attention_map_data = final_attention_map[i, 0, :].cpu(
                ).detach().numpy()

                arr_min = np.min(attention_map_data)
                arr_max = np.max(attention_map_data)

                # Normalize the attention array between 0 and 1 for proper alpha
                # values
                normalized_arr = (attention_map_data
                                  - arr_min) / (arr_max - arr_min)

                # Define a colormap
                cmap = plt.cm.viridis

                # Plot an overlay with a colormap
                axes[i].imshow(normalized_arr[None, :], cmap=cmap, aspect='auto', alpha=0.3, extent=(
                    0, len(channel_data), np.min(channel_data), np.max(channel_data)))

                # Add colorbar
                cbar = plt.colorbar(
                    plt.cm.ScalarMappable(
                        cmap=cmap), ax=axes[i])
                cbar.set_label('Attention')

            axes[i].set_ylim(np.min(channel_data), np.max(channel_data))

        plt.subplots_adjust(top=0.95, bottom=0.05, hspace=0.4)

        # Save to destination path
        # Create the directory if it doesn't exist
        if not os.path.exists(destination):
            os.makedirs(destination)

        # Save the graph image to disk
        path = os.path.join(destination, filename)
        plt.savefig(path)

        # Clean up plt lib
        plt.close()
        plt.clf()
