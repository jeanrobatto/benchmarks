import torch


class LocalizationAttention(torch.nn.Module):
    """
    This class implements the Efficient Localization Attention mechanism.

    Source: https://arxiv.org/html/2403.01123v1
    Section: 3.5

    It applies convolution operations along height and width dimensions separately to compute attention
    weights for each spatial dimension. The resulting attention map is used to modulate the input feature map.

    Adaptation:
        Instead of a single method as proposed in the paper, this class is a torch module for compatibility
        wth the framework. In addition, there is an option to extract the used feature map, which is later used
        for graphing and reporting purposes.

        Lastly, the number of Groups in the GroupNorm operation is set to 1 instead of 16 because the data almost
        always has a prime number of channels.

    Args:
        channel (int): Number of input channels.
        kernel_size (int): Size of the convolution kernel, defaults to 9.

    Attributes:
        conv (torch.nn.Conv1d): Convolutional layer for height and width attention computation.
        gn (torch.nn.GroupNorm): Group normalization layer to normalize the attention map.
        sigmoid (torch.nn.Sigmoid): Sigmoid activation function to constrain attention values between 0 and 1.

    """
    def __init__(self, channel, kernel_size=9):
        """
        Initialize the LocalizationAttention module.

        Args:
            channel (int): Number of input channels.
            kernel_size (int): Size of the convolution kernel, defaults to 9.

        """
        super(LocalizationAttention, self).__init__()

        self.conv = torch.nn.Conv1d(
            channel,
            channel,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channel,
            bias=False
        )

        # Equivalent to LayerNorm since there is only one group
        self.gn = torch.nn.GroupNorm(1, channel)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, save_map=False):
        """
        Forward pass of the Localization Attention mechanism.

        Args:
            x (torch.Tensor): Input feature map of shape (batch_size, channels, height, width).
            save_map (bool): Flag indicating whether to save the attention map, defaults to False.

        Returns:
            torch.Tensor: Modulated feature map with applied attention.
            torch.Tensor or None: Attention map if save_map is True, else None.

        """
        b, c, h, w = x.shape

        # Height in our problem domain is always one, but this code allows this module to be more reusable
        x_h = torch.mean(x, dim=3, keepdim=True).view(b, c, h)
        x_w = torch.mean(x, dim=2, keepdim=True).view(b, c, w)

        x_h = self.sigmoid(self.gn(self.conv(x_h))).view(b, c, h, 1)
        x_w = self.sigmoid(self.gn(self.conv(x_w))).view(b, c, 1, w)

        result = x * x_h * x_w

        if save_map:
            # Save current attention map for future use
            attention_map = torch.cat((x_h, x_w), dim=3)
            return result, attention_map

        return result