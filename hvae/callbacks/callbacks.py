"""Lightning callbacks."""
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback

from hvae.utils.dct import reconstruct_dct
from hvae.visualization import draw_batch, draw_reconstructions


class VisualizationCallback(Callback):
    """Callback for visualizing VAE reconstructions."""

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
            x, y = batch
            pl_module.eval()
            _, x_hat = pl_module.step((x.to(pl_module.device), y.to(pl_module.device)))
            images = draw_reconstructions(
                x.detach().cpu().numpy(), x_hat.detach().cpu().numpy()
            )
            pl_module.logger.log_image("reconstructions", images=[images])

            # visualize batch and its DCT reconstructions for different k
            x, _ = batch
            x_1 = reconstruct_dct(x, k=1)
            x_2 = reconstruct_dct(x, k=2)
            x_4 = reconstruct_dct(x, k=4)
            x_8 = reconstruct_dct(x, k=8)
            images = draw_reconstructions(
                x.detach().cpu().numpy(),
                x1.detach().cpu().numpy(),
                x2.detach().cpu().numpy(),
                x4.detach().cpu().numpy(),
                x8.detach().cpu().numpy(),
            )
            pl_module.logger.log_image("dct_reconstructions", images=[images])

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Visualize model samples."""
        samples = pl_module.sample(25).detach().cpu().numpy()
        images = draw_batch(samples)
        pl_module.logger.log_image("samples", images=[images])


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
