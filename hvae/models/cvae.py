"""A conditional VAE model."""
from hvae.models import VAE

import torch
from torch import Tensor, nn
from torch.nn import functional as F


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

    def forward(self, x: Tensor, y: Tensor) -> list[Tensor]:
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
