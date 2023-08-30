"""Training script."""
import hydra
import torch
import torchvision
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from torchvision import transforms

from hvae.callbacks import LoggingCallback, MetricsCallback, VisualizationCallback
from hvae.models import CVAE, VAE


@hydra.main(config_path="configs", config_name="main")
def train(cfg: DictConfig) -> None:
    """Train a model."""
    torch.set_float32_matmul_precision("high")
    train_dataloader, val_dataloader = get_dataloaders(cfg)
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        save_dir=cfg.wandb.dir,
        config=config,
        group=cfg.wandb.group,
        mode=cfg.wandb.mode,
    )
    trainer = Trainer(
        accelerator="auto",
        default_root_dir=cfg.wandb.dir,
        devices=1,
        enable_checkpointing=False,
        enable_progress_bar=False,
        log_every_n_steps=cfg.wandb.log_every_n_steps,
        logger=wandb_logger,
        max_epochs=cfg.training.max_epochs,
        callbacks=[LoggingCallback(), MetricsCallback(), VisualizationCallback()],
    )
    model = get_model(cfg)
    trainer.fit(model, train_dataloader, val_dataloader)


def get_dataloaders(cfg: DictConfig):
    if cfg.dataset.name != "cifar10":
        raise ValueError(f"Invalid dataset name: {cfg.dataset.name}.")
    dataset = torchvision.datasets.CIFAR10(
        root=cfg.dataset.root,
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    # filter out everything except desired class
    if cfg.dataset.classes is not None:
        dataset = torch.utils.data.Subset(
            dataset,
            [i for i, (_, label) in enumerate(dataset) if label in cfg.dataset.classes],
        )
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [cfg.dataset.train_split, cfg.dataset.val_split]
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )
    return train_dataloader, val_dataloader


def get_model(cfg: DictConfig):
    if cfg.model.name == "vae":
        model = VAE(
            img_size=cfg.dataset.img_size,
            in_channels=cfg.dataset.num_channels,
            channels=cfg.model.channels,
            latent_dim=cfg.model.latent_dim,
            beta=cfg.model.beta,
            lr=cfg.training.lr,
        )
    elif cfg.model.name == "cvae":
        model = CVAE(
            num_classes=cfg.dataset.num_classes,
            img_size=cfg.dataset.img_size,
            in_channels=cfg.dataset.num_channels,
            channels=cfg.model.channels,
            latent_dim=cfg.model.latent_dim,
            beta=cfg.model.beta,
            lr=cfg.training.lr,
        )
    else:
        raise ValueError(f"Invalid model name: {cfg.model.name}.")
    return model


if __name__ == "__main__":
    train()
