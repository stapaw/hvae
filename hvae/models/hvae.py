"""A hierarchical deep convolutional VAE model.
Based on https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_hierarchical_example.ipynb
"""
import copy

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from hvae.models import VAE
from hvae.models.blocks import MLP
from hvae.utils.dct import reconstruct_dct


class HVAE(VAE):
    """Conditional VAE"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nn_r_2 = MLP(
            dims=[
                self.encoder_output_size,
                self.encoder_output_size,
                self.encoder_output_size,
            ]
        )
        self.nn_delta_1 = MLP(
            dims=[
                self.encoder_output_size,
                self.encoder_output_size,
                2 * (self.latent_dim * 2),
            ]
        )
        self.nn_delta_2 = MLP(
            dims=[
                self.encoder_output_size,
                self.encoder_output_size,
                2 * self.latent_dim,
            ]
        )
        self.nn_z_1 = MLP(
            dims=[
                self.latent_dim,
                self.encoder_output_size,
                2 * (self.latent_dim * 2),
            ]
        )
        self.decoder_input = nn.Linear(2 * self.latent_dim, self.encoder_output_size)

    def step(self, batch):
        x, y = batch
        outputs = self.forward(x, y)
        outputs["x"] = x
        loss = self.loss_function(**outputs)
        return loss, outputs["x_hat"]

    def forward(self, x: Tensor, y: Tensor) -> list[Tensor]:
        """Perform the forward pass.
        Args:
            x: Input tensor of shape (B x C x H x W)
        Returns:
            List of tensors [reconstructed input, latent mean, latent log variance]
        """
        r_1 = self.encoder(x).flatten(start_dim=1)  # add an MLP between enc and r_1?
        r_2 = self.nn_r_2(r_1)

        delta_1 = self.nn_delta_1(r_1)
        delta_mu_1, delta_log_var_1 = torch.chunk(delta_1, 2, dim=1)
        delta_log_var_1 = F.hardtanh(delta_log_var_1, -7.0, 2.0)

        delta_2 = self.nn_delta_2(r_2)
        delta_mu_2, delta_log_var_2 = torch.chunk(delta_2, 2, dim=1)
        delta_log_var_2 = F.hardtanh(delta_log_var_2, -7.0, 2.0)
        z_2 = self.reparameterize(delta_mu_2, delta_log_var_2)

        h_1 = self.nn_z_1(z_2)
        mu_1, log_var_1 = torch.chunk(h_1, 2, dim=1)
        z_1 = self.reparameterize(mu_1 + delta_mu_1, log_var_1 + delta_log_var_1)

        x_hat = self.decode(z_1)

        return {
            "x_hat": x_hat,
            "delta_mu_1": delta_mu_1,
            "delta_log_var_1": delta_log_var_1,
            "log_var_1": log_var_1,
            "delta_mu_2": delta_mu_2,
            "delta_log_var_2": delta_log_var_2,
        }

    def loss_function(
        self,
        x: Tensor,
        x_hat: Tensor,
        delta_mu_1: Tensor,
        delta_log_var_1: Tensor,
        log_var_1: Tensor,
        delta_mu_2: Tensor,
        delta_log_var_2: Tensor,
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

        kl_divergence_z1 = 0.5 * (
            delta_mu_1**2 / torch.exp(log_var_1)
            + torch.exp(delta_log_var_1)
            - delta_log_var_1
            - 1
        ).sum(-1)

        kl_divergence_z2 = 0.5 * (
            delta_mu_2**2 + torch.exp(delta_log_var_2) - delta_log_var_2 - 1
        ).sum(-1)

        kl_divergence = kl_divergence_z1.mean() + kl_divergence_z2.mean()

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
        z_2 = torch.randn(num_samples, self.latent_dim).to(self.device)
        h_1 = self.nn_z_1(z_2)
        mu_1, log_var_1 = torch.chunk(h_1, 2, dim=1)
        z_1 = self.reparameterize(mu_1, log_var_1)
        return self.decode(z_1)


class DCTHVAE(HVAE):
    def __init__(self, k: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.decoder_dct = copy.deepcopy(self.decoder)
        self.decoder_input_dct = nn.Linear(self.latent_dim, self.encoder_output_size)
        self.k = k

    def step(self, batch):
        x, y = batch
        outputs = self.forward(x, y)
        outputs["x"] = x
        loss = self.loss_function(**outputs)
        return loss, outputs["x_hat"], outputs["x_hat_dct"]

    def forward(self, x: Tensor, y: Tensor) -> list[Tensor]:
        """Perform the forward pass.
        Args:
            x: Input tensor of shape (B x C x H x W)
        Returns:
            List of tensors [reconstructed input, latent mean, latent log variance]
        """
        r_1 = self.encoder(x).flatten(start_dim=1)  # add an MLP between enc and r_1?
        r_2 = self.nn_r_2(r_1)

        delta_1 = self.nn_delta_1(r_1)
        delta_mu_1, delta_log_var_1 = torch.chunk(delta_1, 2, dim=1)
        delta_log_var_1 = F.hardtanh(delta_log_var_1, -7.0, 2.0)

        delta_2 = self.nn_delta_2(r_2)
        delta_mu_2, delta_log_var_2 = torch.chunk(delta_2, 2, dim=1)
        delta_log_var_2 = F.hardtanh(delta_log_var_2, -7.0, 2.0)
        z_2 = self.reparameterize(delta_mu_2, delta_log_var_2)
        x_hat_dct = self.decode_dct(z_2)

        h_1 = self.nn_z_1(z_2)
        mu_1, log_var_1 = torch.chunk(h_1, 2, dim=1)
        z_1 = self.reparameterize(mu_1 + delta_mu_1, log_var_1 + delta_log_var_1)

        x_hat = self.decode(z_1)

        return {
            "x_hat": x_hat,
            "x_hat_dct": x_hat_dct,
            "delta_mu_1": delta_mu_1,
            "delta_log_var_1": delta_log_var_1,
            "log_var_1": log_var_1,
            "delta_mu_2": delta_mu_2,
            "delta_log_var_2": delta_log_var_2,
        }

    def decode_dct(self, z_2: Tensor) -> Tensor:
        """Given a latent vector z, return the corresponding image.
        Args:
            z: Latent vector of shape (B x D)
        Returns:
            Tensor of shape (B x C x H x W)
        """
        z = self.decoder_input_dct(z_2)
        z = z.view(
            -1,
            self.channels[-1],
            self.encoder_output_img_size,
            self.encoder_output_img_size,
        )
        return self.decoder_dct(z)

    def loss_function(
        self,
        x: Tensor,
        x_hat: Tensor,
        x_hat_dct: Tensor,
        delta_mu_1: Tensor,
        delta_log_var_1: Tensor,
        log_var_1: Tensor,
        delta_mu_2: Tensor,
        delta_log_var_2: Tensor,
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
        x_dct = reconstruct_dct(x, k=self.k)
        reconstruction_loss = F.mse_loss(x_hat, x, reduction="sum") / x.shape[0]
        reconstruction_loss_dct = (
            F.mse_loss(x_hat_dct, x_dct, reduction="sum") / x_dct.shape[0]
        )

        kl_divergence_z1 = 0.5 * (
            delta_mu_1**2 / torch.exp(log_var_1)
            + torch.exp(delta_log_var_1)
            - delta_log_var_1
            - 1
        ).sum(-1)

        kl_divergence_z2 = 0.5 * (
            delta_mu_2**2 + torch.exp(delta_log_var_2) - delta_log_var_2 - 1
        ).sum(-1)

        kl_divergence = kl_divergence_z1.mean() + kl_divergence_z2.mean()

        loss = reconstruction_loss + reconstruction_loss_dct + self.beta * kl_divergence

        return {
            "loss": loss,
            "reconstruction_loss": reconstruction_loss,
            "reconstruction_loss_dct": reconstruction_loss_dct,
            "kl_divergence": kl_divergence,
        }
