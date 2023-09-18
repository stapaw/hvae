"""Lightning callbacks."""
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
import torch

from hvae.utils.dct import reconstruct_dct
from hvae.visualization import draw_batch, draw_reconstructions


class VisualizationCallback(Callback):
    """Callback for visualizing VAE reconstructions."""

    def __init__(self):
        super().__init__()
        self._logged_dct = False

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """Visualize the first batch and reconstructions."""
        if batch_idx == 0:
            self.log_reconstructions(pl_module, batch, "train")

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """Visualize the first batch and reconstructions."""
        if batch_idx == 0:
            self.log_reconstructions(pl_module, batch, "val")

    @torch.no_grad()
    def log_reconstructions(self, pl_module: pl.LightningModule, batch, stage: str):
        is_training = pl_module.training
        pl_module.eval()
        x, y = batch
        _, *x_hat = pl_module.step((x.to(pl_module.device), y.to(pl_module.device)))
        x_hat = [x.detach().cpu().numpy() for x in x_hat]
        if len(x_hat) == 1:
            images = draw_reconstructions(x.detach().cpu().numpy(), x_hat[0])
        elif len(x_hat) == 2:
            x_dct = reconstruct_dct(x, k=pl_module.k).detach().cpu().numpy()
            images = draw_reconstructions(
                x.detach().cpu().numpy(), x_hat[0], x_dct, x_hat[1]
            )
        pl_module.logger.log_image(f"{stage}/reconstructions", images=[images])

        if not self._logged_dct:
            reconstructions = [
                reconstruct_dct(x, k=k).detach().cpu().numpy() for k in [32, 16, 8, 4]
            ]
            images = draw_reconstructions(x.detach().cpu().numpy(), *reconstructions)
            pl_module.logger.log_image("dct_reconstructions", images=[images])
            self._logged_dct = True
        pl_module.train(is_training)

    @torch.no_grad()
    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Visualize model samples."""
        samples = [
            pl_module.sample(10, level=i).detach().cpu().numpy()
            for i in range(pl_module.num_levels)
        ]
        images = draw_reconstructions(*samples)
        pl_module.logger.log_image("train/samples", images=[images])


class MetricsCallback(Callback):
    """Callback for logging metrics."""

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """Log the training loss."""
        trainer.logger.log_metrics({f"train/{k}": v.item() for k, v in outputs.items()})

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """Log the validation loss."""
        trainer.logger.log_metrics({f"val/{k}": v.item() for k, v in outputs.items()})

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """Log the test loss."""
        trainer.logger.log_metrics({f"test/{k}": v.item() for k, v in outputs.items()})


class LoggingCallback(Callback):
    """Callback for additional logging."""

    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        print(f"Number of batches: {len(trainer.train_dataloader)}.")
        print(f"Number of samples: {len(trainer.train_dataloader.dataset)}.")

    def on_validation_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        print(f"Number of batches: {len(trainer.val_dataloaders)}.")
        print(f"Number of samples: {len(trainer.val_dataloaders.dataset)}.")

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        print(f"Number of batches: {len(trainer.test_dataloaders)}.")
        print(f"Number of samples: {len(trainer.test_dataloaders.dataset)}.")
