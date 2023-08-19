"""Lightning modules for the models."""
from typing import List, Optional

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torchmetrics import R2Score

from codis.models import MLP, BetaVAE
from codis.models.blocks import Encoder
from codis.utils import to_numpy
from codis.data import Latents
from codis.visualization import draw_batch_and_reconstructions

# pylint: disable=arguments-differ,unused-argument,too-many-ancestors


class VAE(pl.LightningModule):
    """The β-VAE Lightning module."""

    def __init__(
        self,
        img_size: int = 64,
        in_channels: int = 1,
        latent_dim: int = 10,
        channels: Optional[list] = None,
        beta: float = 1.0,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.model = BetaVAE(
            img_size,
            in_channels,
            channels,
            latent_dim,
            beta,
        )
        self.save_hyperparameters()
        self.lr = lr

    @property
    def latent_dim(self):
        """Dimensionality of the latent space."""
        return self.model.latent_dim

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Perform the forward pass."""
        return self.model(x)

    def loss_function(self, *args, **kwargs):
        """Calculate the loss."""
        return self.model.loss_function(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        # sourcery skip: class-extract-method
        """Perform a training step."""
        loss = self._step(batch)
        self.log_dict({f"{k}_vae_train": v for k, v in loss.items()})
        return loss["loss"]

    def validation_step(self, batch, batch_idx):
        """Perform a validation step."""
        loss = self._step(batch)
        self.log_dict({f"{k}_vae_val": v for k, v in loss.items()})
        if batch_idx == 0:
            x, _ = batch
            self._log_reconstructions(x)
        return loss["loss"]

    def _step(self, batch):
        """Perform a training or validation step."""
        x, _ = batch
        x_hat, mu, log_var = self.forward(x)
        return self.model.loss_function(x, x_hat, mu, log_var)

    def _log_reconstructions(self, x):
        """Log reconstructions alongside original images."""
        x_hat, _, _ = self.forward(x)
        reconstructions = draw_batch_and_reconstructions(to_numpy(x), to_numpy(x_hat))
        self.logger.log_image("reconstructions", images=[reconstructions])

    def configure_optimizers(self):
        """Initialize the optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.lr)