"""Training script."""
import hydra
import torch
import torchvision
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig
from torch import transforms

from hvae.callbacks import LoggingCallback, VisualizationCallback
from hvae.models import VAE


@hydra.main(config_path="configs", config_name="main")
def train(cfg: DictConfig) -> None:
    """Train a model."""
    if cfg.dataset.name == "cifar10":
        dataset = torchvision.datasets.CIFAR10(
            root=cfg.dataset.root,
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.num_workers,
    )
    wandb_logger = WandbLogger(project=cfg.wandb.project, save_dir=cfg.wandb.dir)
    wandb_logger.experiment.config.update(cfg)
    trainer = Trainer(
        accelerator="auto",
        default_root_dir=cfg.wandb.dir,
        devices=1,
        enable_checkpointing=False,
        enable_progress_bar=False,
        log_every_n_steps=cfg.wandb.log_every_n_steps,
        logger=wandb_logger,
        max_epochs=cfg.training.max_epochs,
        callbacks=[LoggingCallback(), VisualizationCallback()],
    )
    if cfg.model.name == "vae":
        model = VAE(
            img_size=cfg.dataset.img_size,
            in_channels=cfg.dataset.num_channels,
            channels=cfg.model.channels,
            latent_dim=cfg.model.latent_dim,
            beta=cfg.model.beta,
            lr=cfg.training.lr,
        )

    trainer.fit(model, dataloader)


if __name__ == "__main__":
    train()
