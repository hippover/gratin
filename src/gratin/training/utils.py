import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from .callbacks import LatentSpaceSaver, Plotter
from ..models.main_net import MainNet
from ..data.datamodule import DataModule


def setup_model_and_dm(tasks, ds_params, net_params, dl_params, graph_info):
    pl.seed_everything(1234)
    dm = DataModule(ds_params=ds_params, dl_params=dl_params, graph_info=graph_info)
    dm.setup()
    model = MainNet(
        tasks=tasks, latent_dim=net_params["latent_dim"], n_c=net_params["n_c"], dm=dm
    )

    return model, dm


def setup_trainer(logger, dirpath="/gaia/models", tag="default"):
    ES = EarlyStopping(
        monitor="train_loss",  # We do not care about overfitting, train loss is more stable than validation
        min_delta=0.001,
        patience=5,
        verbose=True,
        mode="min",
        strict=True,
    )
    LRM = LearningRateMonitor("epoch")
    CKPT = ModelCheckpoint(
        dirpath=dirpath, filename=tag, monitor="train_loss", verbose=True, mode="min"
    )
    PLT = Plotter()

    trainer = pl.Trainer(
        auto_select_gpus=True,
        gradient_clip_val=1.0,
        reload_dataloaders_every_n_epochs=1,
        callbacks=[ES, LRM, CKPT, PLT],
        log_every_n_steps=150,
        max_epochs=100,
        detect_anomaly=True,
        track_grad_norm=2,
        logger=logger,
    )
    return trainer
