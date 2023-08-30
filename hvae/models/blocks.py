"""Common blocks used in models."""
from torch import nn
from typing import Callable


class Encoder(nn.Module):
    """A simple encoder model."""

    def __init__(
        self, channels: list[int], in_channels: int = 1, activation: Callable = nn.GELU
    ) -> None:
        """Initialize the encoder.
        Args:
            in_channels: The number of input channels.
            channels: The number of channels in each hidden layer.
        Returns:
            None
        """
        super().__init__()
        channels = [in_channels] + channels
        module = [
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                activation(),
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.BatchNorm2d(out_channels),
                activation(),
            )
            for in_channels, out_channels in zip(channels[:-1], channels[1:])
        ]
        self.encoder = nn.Sequential(*module)

    def forward(self, x):
        """Forward pass of the encoder."""
        x = self.encoder(x)
        return x


class Decoder(nn.Module):
    """A simple decoder model."""

    def __init__(
        self,
        channels: list[int],
        out_channels: int = 1,
        activation: Callable = nn.GELU,
        last_activation: Callable = nn.Sigmoid,
    ) -> None:
        """Initialize the decoder.
        Args:
            out_channels: The number of output channels.
            hidden_dims: The number of channels in each hidden layer.
        Returns:
            None
        """
        super().__init__()
        modules = [
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                activation(),
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.BatchNorm2d(out_channels),
                activation(),
            )
            for in_channels, out_channels in zip(channels[:-1], channels[1:])
        ]
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    channels[-1],
                    channels[-1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.BatchNorm2d(channels[-1]),
                activation(),
                nn.Conv2d(
                    channels[-1],
                    out_channels,
                    kernel_size=3,
                    padding=1,
                ),
                last_activation(),
            )
        )
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        """Forward pass of the decoder."""
        x = self.model(x)
        return x


class MLP(nn.Module):
    """A simple multi-layer perceptron.
    Args:
        dims: A list of dimensions for each layer, including input and output
        dropout_rate: The dropout rate
        activation: The activation function for the hidden layers
        last_activation: The activation function for the last layer
    """

    def __init__(
        self,
        dims: list[int],
        dropout_rate: float = 0.0,
        activation: Callable = nn.GELU,
        last_activation: Callable = nn.Identity,
    ):
        super().__init__()
        modules = [
            nn.Sequential(
                nn.Linear(n_in, n_out), activation(), nn.Dropout(dropout_rate)
            )
            for n_in, n_out in zip(dims[:-2], dims[1:-1])
        ]
        modules.append(nn.Sequential(nn.Linear(dims[-2], dims[-1]), last_activation()))
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        """Forward pass of the MLP."""
        return self.model(x)
