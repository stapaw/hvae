"""Lightning callbacks."""
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback

from hvae.visualization import draw_batch, draw_reconstructions


class VisualizationCallback(Callback):
    """Callback for visualizing VAE reconstructions."""

    def __init__(self):
        super().__init__()

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
            x, _ = batch
            pl_module.eval()
            x_hat, *_ = pl_module(x.to(pl_module.device))
            images = draw_reconstructions(
                x.detach().cpu().numpy(), x_hat.detach().cpu().numpy()
            )
            pl_module.logger.log_image("reconstructions", images=[images])

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Visualize model samples."""
        samples = pl_module.sample(25).detach().cpu().numpy()
        images = draw_batch(samples)
        pl_module.logger.log_image("samples", images=[images])


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
