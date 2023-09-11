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
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding='same',
                ),
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
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding='same',
                ),
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
        activation: Callable = nn.GELU,
        last_activation: Callable = nn.Identity,
    ):
        super().__init__()
        modules = [
            nn.Sequential(nn.Linear(n_in, n_out), activation(), nn.BatchNorm1d(n_out))
            for n_in, n_out in zip(dims[:-2], dims[1:-1])
        ]
        modules.append(nn.Sequential(nn.Linear(dims[-2], dims[-1]), last_activation()))
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        """Forward pass of the MLP."""
        return self.model(x)


class ResBlock(nn.Module):
    """A two-convolutional layer residual block."""

    def __init__(self, c_in, c_out, k, s=1, p=1, mode="encode"):
        assert mode in ["encode", "decode"], "Mode must be either 'encode' or 'decode'."
        super(ResBlock, self).__init__()
        if mode == "encode":
            self.conv1 = nn.Conv2d(c_in, c_out, k, s, p)
            self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1)
        elif mode == "decode":
            self.conv1 = nn.ConvTranspose2d(c_in, c_out, k, s, p)
            self.conv2 = nn.ConvTranspose2d(c_out, c_out, 3, 1, 1)
        self.relu = nn.ReLU()
        self.BN = nn.BatchNorm2d(c_out)
        self.resize = s > 1 or (s == 1 and p == 0) or c_out != c_in

    def forward(self, x):
        conv1 = self.BN(self.conv1(x))
        relu = self.relu(conv1)
        conv2 = self.BN(self.conv2(relu))
        if self.resize:
            x = self.BN(self.conv1(x))
        return self.relu(x + conv2)


class ResEncoder(nn.Module):
    """
    Encoder class, mainly consisting of three residual blocks.
    """

    def __init__(self):
        super(ResEncoder, self).__init__()
        self.init_conv = nn.Conv2d(3, 16, 3, 1, 1)  # 16 32 32
        self.BN = nn.BatchNorm2d(16)
        self.rb1 = ResBlock(16, 16, 3, 2, 1, "encode")  # 16 16 16
        self.rb2 = ResBlock(16, 32, 3, 1, 1, "encode")  # 32 16 16
        self.rb3 = ResBlock(32, 32, 3, 2, 1, "encode")  # 32 8 8
        self.rb4 = ResBlock(32, 48, 3, 1, 1, "encode")  # 48 8 8
        self.rb5 = ResBlock(48, 48, 3, 2, 1, "encode")  # 48 4 4
        self.rb6 = ResBlock(48, 64, 3, 2, 1, "encode")  # 64 2 2
        self.relu = nn.ReLU()

    def forward(self, inputs):
        init_conv = self.relu(self.BN(self.init_conv(inputs)))
        rb1 = self.rb1(init_conv)
        rb2 = self.rb2(rb1)
        rb3 = self.rb3(rb2)
        rb4 = self.rb4(rb3)
        rb5 = self.rb5(rb4)
        rb6 = self.rb6(rb5)
        return rb6


class ResDecoder(nn.Module):
    """
    Decoder class, mainly consisting of two residual blocks.
    """

    def __init__(self):
        super(ResDecoder, self).__init__()
        self.rb1 = ResBlock(64, 48, 2, 2, 0, "decode")  # 48 4 4
        self.rb2 = ResBlock(48, 48, 2, 2, 0, "decode")  # 48 8 8
        self.rb3 = ResBlock(48, 32, 3, 1, 1, "decode")  # 32 8 8
        self.rb4 = ResBlock(32, 32, 2, 2, 0, "decode")  # 32 16 16
        self.rb5 = ResBlock(32, 16, 3, 1, 1, "decode")  # 16 16 16
        self.rb6 = ResBlock(16, 16, 2, 2, 0, "decode")  # 16 32 32
        self.out_conv = nn.ConvTranspose2d(16, 3, 3, 1, 1)  # 3 32 32
        self.tanh = nn.Tanh()

    def forward(self, inputs):
        rb1 = self.rb1(inputs)
        rb2 = self.rb2(rb1)
        rb3 = self.rb3(rb2)
        rb4 = self.rb4(rb3)
        rb5 = self.rb5(rb4)
        rb6 = self.rb6(rb5)
        out_conv = self.out_conv(rb6)
        return self.tanh(out_conv)
