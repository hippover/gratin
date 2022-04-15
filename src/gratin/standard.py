from gratin.models.main_net import MainNet
from gratin.data.datamodule import DataModule
import pytorch_lightning as pl
from gratin.models.utils import get_predictions_of_dl
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
import os
from umap import ParametricUMAP
import tensorflow as tf
import numpy as np
import logging
import torch.cuda

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

graph_info = {
    "edges_per_point": 10,
    "clip_trajs": False,
    "scale_types": ["step_sum", "pos_std", "step_std"],
    "log_features": True,
    "data_type": "no_features",  # no features because features are all computed by the model
    "edge_method": "geom_causal",
}


def train_model(
    export_path: str,
    time_delta: float,
    max_n_epochs: int = 100,
    num_workers: int = 0,  # number of workers during the training process (should be < # CPUs)
    dim: int = 2,  # Dimension of trajectories. 2 or 3
    log_diffusion_range: tuple = (
        -2.0,
        1.1,
    ),  # log-diffusion is drawn following a truncated centered gaussian in this range
    length_range: tuple = (7, 35),  # length is drawn in a log-uniform way
    noise_range: tuple = (
        0.015,
        0.05,
    ),  # localization uncertainty, in micrometers (one value per trajectory)
):

    if not os.path.exists(export_path):
        os.mkdir(export_path)
    assert os.path.isdir(export_path)

    dl_params = {"batch_size": 128, "num_workers": num_workers}

    ds_params = {
        "dim": dim,  # can be (1, 2 or 3)
        "RW_types": [
            "fBM",
            "LW",
            "sBM",
            "OU",
            "CTRW",
        ],  # Types of random walks used during training
        "time_delta": time_delta,
        "logdiffusion_range": log_diffusion_range,
        "length_range": length_range,
        "noise_range": noise_range,
    }

    model = MainNet(
        tasks=["alpha", "model"],
        n_c=16,
        latent_dim=16,
        lr=1e-3,
        dim=2,
        RW_types=ds_params["RW_types"],
        scale_types=["step_std", "pos_std"],
    )

    pl.seed_everything(1)
    dm = DataModule(ds_params=ds_params, dl_params=dl_params, graph_info=graph_info)
    dm.setup(plot=False)

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
        dirpath=export_path,
        filename="model",
        monitor="train_loss",
        verbose=True,
        mode="min",
    )

    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir=os.path.join(export_path, "tb_logs"),
        default_hp_metric=False,
        name="model",
        flush_secs=60,
    )

    trainer = pl.Trainer(
        auto_select_gpus=torch.cuda.is_available(),
        gpus=1 * torch.cuda.is_available(),
        gradient_clip_val=1.0,
        reload_dataloaders_every_n_epochs=1,
        callbacks=[ES, LRM, CKPT],
        log_every_n_steps=150,
        max_epochs=max_n_epochs,
        detect_anomaly=True,
        track_grad_norm=2,
        logger=tb_logger,
    )

    trainer.fit(model=model, datamodule=dm)

    results = trainer.test(model, dm.test_dataloader())
    print(results)

    model.eval()
    output, target, h = get_predictions_of_dl(
        model, dm.test_dataloader(no_parallel=True), latent_samples=int(3e4)
    )

    umap_params = {
        "min_dist": 0.5,
        "n_components": 2,
        "verbose": True,
        "n_neighbors": 200,
        "metric": "cosine",
    }

    encoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(16)),
            tf.keras.layers.Dense(units=256),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Dense(units=16),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Dense(units=2),
        ]
    )

    U = ParametricUMAP(**umap_params, encoder=encoder)
    U.fit(h)

    encoder.save(os.path.join(export_path, "umap"))
    X = np.array(encoder(h))

    alpha = output["alpha"]
    alpha_true = target["alpha"]
    true_model = target["model"]

    plt.figure()
    cond = alpha_true > 0
    cm = plt.scatter(X[cond, 0], X[cond, 1], c=target["log_diffusion"][cond], s=0.1)
    plt.colorbar(cm, label="$\\log_{10}(D)$")
    plt.tight_layout()

    plt.figure()
    cond = alpha_true > 0
    cm = plt.scatter(X[cond, 0], X[cond, 1], c=alpha_true[cond], s=0.1)
    plt.colorbar(cm, label="$\\alpha$")
    plt.tight_layout()

    plt.figure()
    for m in np.unique(true_model):
        plt.scatter(
            X[true_model == m, 0],
            X[true_model == m, 1],
            s=1,
            label=ds_params["RW_types"][m],
        )
    plt.legend(markerscale=10)
    plt.tight_layout()

    plt.figure()
    mean_err = [
        np.mean(
            np.abs(
                alpha[(target["length"] == L) & (alpha_true > 0)]
                - alpha_true[(target["length"] == L) & (alpha_true > 0)]
            )
        )
        for L in np.unique(target["length"])
    ]
    length = np.unique(target["length"])
    plt.plot(length, mean_err)
    plt.ylabel("MAE($\\alpha$)")
    plt.xlabel("Trajectory length")
    plt.tight_layout()

    pass
