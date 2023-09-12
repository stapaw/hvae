import numpy as np
import torch
from scipy.fft import dctn, idctn
from torch.utils.data import Dataset
from hvae.utils.dct import get_mask

from pathlib import Path

import hydra
import torch
import torchvision
from omegaconf import DictConfig, OmegaConf
from torchvision import transforms

from hvae.callbacks import LoggingCallback, MetricsCallback, VisualizationCallback

class CustomDataset(Dataset):
    def __init__(self, img_tensors, img_labels, img_transform=None, target_transform=None):
        self.img_labels = img_labels
        self.imgs = img_tensors
        self.transform = img_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.imgs[idx]
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def get_pretrain_dataset(n, k, means, stds):
    data = []
    for _ in range(0, n):
        x = torch.normal(means, stds) * get_mask(k, (1, 32, 32))
        w = idctn(np.array(x)).astype(float)
        data.append(w.copy())
    data = torch.stack([torch.from_numpy(a) for a in data]).type(torch.FloatTensor)
    return data, torch.zeros(n)


@hydra.main(config_path="configs", config_name="main")
def train(cfg: DictConfig) -> None:
    """Train a model."""
    torch.set_float32_matmul_precision("high")
    train_dataloader, val_dataloader = get_dataloaders(cfg)
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=[LoggingCallback(), MetricsCallback(), VisualizationCallback()],
        logger={"config": config},
    )
    model = hydra.utils.instantiate(cfg.model)
    trainer.fit(model, train_dataloader, val_dataloader)


def get_dataloaders(cfg: DictConfig):
    # root = Path(hydra.utils.get_original_cwd()) / Path(cfg.dataset.root)

    means = torch.load("/home/spawlak/hvae/pretrain/means.pt")
    stds = torch.load("/home/spawlak/hvae/pretrain/stds.pt")

    pre_ds = CustomDataset(
        *get_pretrain_dataset(100000, cfg.model.k, means, stds))


    # if cfg.dataset.name != "cifar10":
    #     raise ValueError(f"Invalid dataset name: {cfg.dataset.name}.")
    # dataset = torchvision.datasets.CIFAR10(
    #     root=root,
    #     train=True,
    #     download=True,
    #     transform=transforms.ToTensor(),
    # )
    # # filter out everything except desired class
    # if cfg.dataset.classes is not None:
    #     dataset = torch.utils.data.Subset(
    #         dataset,
    #         [i for i, (_, label) in enumerate(dataset) if label in cfg.dataset.classes],
    #     )
    train_dataset, val_dataset = torch.utils.data.random_split(
        pre_ds, [cfg.dataset.train_split, cfg.dataset.val_split]
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

if __name__ == "__main__":
    train()

