"""The β-VAE model. See https://openreview.net/forum?id=Sy2fzU9gl for details."""
from typing import List, Optional

import lightning.pytorch as pl
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from hvae.models.blocks import Decoder, Encoder


class VAE(pl.LightningModule):
    """The β-VAE model class."""

    def __init__(
        self,
        img_size: int = 64,
        in_channels: int = 1,
        channels: Optional[list] = None,
        latent_dim: int = 10,
        beta: float = 0.0,
        lr: float = 1e-3,
    ) -> None:
        """Initialize the model.
        Args:
            img_size: Size of the input image in pixels
            in_channels: Number of input channels
            channels: Number of channels in the encoder and decoder networks
            latent_dim: Latent space dimensionality
            beta: Weight of the KL divergence loss term
            lr: Learning rate
        Returns:
            None
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        self.lr = lr

        if channels is None:
            channels = [4, 4, 8, 8, 16]
        self.channels = channels

        self.encoder_output_size = img_size // 2 ** len(channels)
        assert self.encoder_output_size > 0, "Too many layers for the input size."
        encoder_output_dim = self.encoder_output_size**2 * channels[-1]

        self.encoder = Encoder(channels, in_channels)
        self.fc_mu = nn.Linear(encoder_output_dim, latent_dim)
        self.fc_var = nn.Linear(encoder_output_dim, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, encoder_output_dim)
        self.decoder = Decoder(list(reversed(channels)), in_channels)

    def configure_optimizers(self):
        """Configure the optimizers."""
        return torch.optim.Adam(
            [
                {"params": self.encoder.parameters(), "lr": self.lr},
                {"params": self.fc_mu.parameters(), "lr": self.lr},
                {"params": self.fc_var.parameters(), "lr": self.lr},
                {"params": self.decoder_input.parameters(), "lr": self.lr},
                {"params": self.decoder.parameters(), "lr": self.lr},
            ]
        )

    def training_step(self, batch, batch_idx):
        """Perform a training step."""
        return self.step(batch)[0]

    def validation_step(self, batch, batch_idx):
        """Perform a validation step."""
        return self.step(batch)[0]

    def test_step(self, batch, batch_idx):
        """Perform a test step."""
        return self.step(batch)[0]

    def step(self, batch):
        x, _ = batch
        x_hat, mu, log_var = self.forward(x)
        loss = self.loss_function(x, x_hat, mu, log_var)
        return loss, x_hat

    def forward(self, x: Tensor) -> List[Tensor]:
        """Perform the forward pass.
        Args:
            x: Input tensor of shape (B x C x H x W)
        Returns:
            List of tensors [reconstructed input, latent mean, latent log variance]
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        return x_hat, mu, log_var

    def encode(self, x: Tensor) -> List[Tensor]:
        """Pass the input through the encoder network and return the latent code.
        Args:
            x: Input tensor of shape (B x C x H x W)
        Returns:
            List of latent codes
        """
        x = self.encoder(x).flatten(start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Perform the reparameterization trick.
        Args:
            mu: Mean of the latent Gaussian of shape (N x D)
            logvar: Standard deviation of the latent Gaussian of shape (N x D)
        Returns:
            Sampled latent vector (N x D)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor) -> Tensor:
        """Pass the latent code through the decoder network and return the reconstructed input.
        Args:
            z: Latent code tensor of shape (B x D)
        Returns:
            Reconstructed input of shape (B x C x H x W)
        """
        z = self.decoder_input(z)
        z = z.view(
            -1,
            self.channels[-1],
            self.encoder_output_size,
            self.encoder_output_size,
        )
        return self.decoder(z)

    def loss_function(
        self,
        x: Tensor,
        x_hat: Tensor,
        mu: Tensor,
        log_var: Tensor,
    ) -> dict:
        """Compute the loss given ground truth images and their reconstructions.
        Args:
            x: Ground truth images of shape (B x C x H x W)
            x_hat: Reconstructed images of shape (B x C x H x W)
            mu: Latent mean of shape (B x D)
            log_var: Latent log variance of shape (B x D)
            kld_weight: Weight for the Kullback-Leibler divergence term
        Returns:
            Dictionary containing the loss value and the individual losses
        """
        reconstruction_loss = F.mse_loss(x_hat, x, reduction="sum") / x.shape[0]

        kl_divergence = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )

        loss = reconstruction_loss + self.beta * kl_divergence

        return {
            "loss": loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_divergence": kl_divergence,
        }

    def sample(self, num_samples: int) -> Tensor:
        """Sample a vector in the latent space and return the corresponding image.
        Args:
            num_samples: Number of samples to generate
            current_device: Device to run the model
        Returns:
            Tensor of shape (num_samples x C x H x W)
        """
        z = torch.randn(num_samples, self.latent_dim).to(self.device)
        return self.decode(z)

    def reconstruct(self, x: Tensor, **kwargs) -> Tensor:
        """Given an input image x, returns the reconstructed image.
        Args:
            x: Input tensor of shape (B x C x H x W)
        Returns:
            Reconstructed input of shape (B x C x H x W)
        """
        return self.forward(x)[0]


class CVAE(VAE):
    """Conditional VAE"""

    def __init__(self, num_classes: int = None, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.fc_z = nn.Linear(self.latent_dim + num_classes, self.latent_dim)

    def step(self, batch):
        x, y = batch
        x_hat, mu, log_var = self.forward(x, y)
        loss = self.loss_function(x, x_hat, mu, log_var)
        return loss, x_hat

    def forward(self, x: Tensor, y: Tensor) -> List[Tensor]:
        """Perform the forward pass.
        Args:
            x: Input tensor of shape (B x C x H x W)
        Returns:
            List of tensors [reconstructed input, latent mean, latent log variance]
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z, y)
        return x_hat, mu, log_var

    def decode(self, z: Tensor, y: Tensor) -> Tensor:
        """Pass the latent code through the decoder network and return the reconstructed input.
        Args:
            z: Latent code tensor of shape (B x D)
            y: Class label tensor of shape (B x N)
        Returns:
            Reconstructed input of shape (B x C x H x W)
        """
        y = F.one_hot(y, num_classes=self.num_classes).float().to(self.device)
        z = torch.cat([z, y], dim=1)
        z = self.fc_z(z)
        return super().decode(z)

    def sample(self, num_samples: int, y: Tensor = None) -> Tensor:
        """Sample a vector in the latent space and return the corresponding image.
        Args:
            num_samples: Number of samples to generate
            y: Class label tensor of shape (num_samples x 1)
        Returns:
            Tensor of shape (num_samples x C x H x W)
        """
        if y is None:
            y = torch.randint(self.num_classes, size=(num_samples,)).to(self.device)
        z = torch.randn(num_samples, self.latent_dim).to(self.device)
        return self.decode(z, y)
