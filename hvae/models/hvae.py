"""A hierarchical deep convolutional VAE model.
Based on https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_hierarchical_example.ipynb
"""

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
        x, _ = batch
        outputs = self.forward(x)
        outputs["x"] = x
        loss = self.loss_function(**outputs)
        return loss, outputs["x_hat"]

    def forward(self, x: Tensor) -> list[Tensor]:
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
        z_1 = self.decoder_input(z_1)
        x_hat = self.decode(z_1)

        return {
            "x_hat": x_hat,
            "delta_mu_1": delta_mu_1,
            "delta_log_var_1": delta_log_var_1,
            "log_var_1": log_var_1,
            "delta_mu_2": delta_mu_2,
            "delta_log_var_2": delta_log_var_2,
            "z_2": z_2,
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
        **kwargs,
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

        kl_divergence_z_1 = 0.5 * (
            delta_mu_1**2 / torch.exp(log_var_1)
            + torch.exp(delta_log_var_1)
            - delta_log_var_1
            - 1
        ).sum(-1)

        kl_divergence_z_2 = 0.5 * (
            delta_mu_2**2 + torch.exp(delta_log_var_2) - delta_log_var_2 - 1
        ).sum(-1)

        kl_divergence = kl_divergence_z_1.mean() + kl_divergence_z_2.mean()

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
        z_1 = self.decoder_input(z_1)
        return self.decode(z_1)


class DCTHVAE(HVAE):
    def __init__(self, k: int = 16, gamma=0.5, **kwargs):
        super().__init__(**kwargs)
        self.decoder_input_dct = nn.Linear(self.latent_dim, self.encoder_output_size)
        self.k = k
        self.gamma = gamma

    def step(self, batch):
        x, _ = batch
        outputs = self.forward(x)
        outputs["x"] = x
        loss = self.loss_function(**outputs)
        return loss, outputs["x_hat"], outputs["x_hat_dct"]

    def forward(self, x: Tensor) -> list[Tensor]:
        """Perform the forward pass.
        Args:
            x: Input tensor of shape (B x C x H x W)
        Returns:
            List of tensors [reconstructed input, latent mean, latent log variance]
        """
        outputs = super().forward(x)
        z_2 = self.decoder_input_dct(outputs["z_2"])
        x_hat_dct = self.decode(z_2)
        outputs["x_hat_dct"] = x_hat_dct
        return outputs

    def loss_function(self, **kwargs) -> dict:
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
        loss = super().loss_function(**kwargs)
        x_hat_dct = kwargs["x_hat_dct"]
        x_dct = reconstruct_dct(kwargs["x"], k=self.k).to(self.device)
        reconstruction_loss_dct = (
            F.mse_loss(x_hat_dct, x_dct, reduction="sum") / x_dct.shape[0]
        )
        loss["reconstruction_loss_dct"] = reconstruction_loss_dct
        loss["loss"] += (self.gamma - 1) * loss["reconstruction_loss"] + (1-self.gamma) * reconstruction_loss_dct
        return loss

    def sample(self, num_samples: int, level=0) -> Tensor:
        """Sample a vector in the latent space and return the corresponding image.
        Args:
            num_samples: Number of samples to generate
            current_device: Device to run the model
        Returns:
            Tensor of shape (num_samples x C x H x W)
        """
        z_2 = torch.randn(num_samples, self.latent_dim).to(self.device)
        if level == 1:
            z_2 = self.decoder_input_dct(z_2)
            return self.decode(z_2)
        elif level == 0:
            h_1 = self.nn_z_1(z_2)
            mu_1, log_var_1 = torch.chunk(h_1, 2, dim=1)
            z_1 = self.reparameterize(mu_1, log_var_1)
            z_1 = self.decoder_input(z_1)
            return self.decode(z_1)
        raise ValueError(f"Invalid level: {level}.")
