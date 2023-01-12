from gratin.models.main_net import MainNet
from gratin.data.datamodule import DataModule
import pytorch_lightning as pl
from gratin.models.utils import get_predictions_of_dl
from gratin.training.callbacks import Plotter
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from gratin.data.dataset import ExpTrajDataSet
from torch_geometric.loader import DataLoader
import os
from umap import ParametricUMAP
import tensorflow as tf
import numpy as np
import logging
import torch.cuda
import pandas as pd
from typing import Union, List
import warnings

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

graph_info = {
    "edges_per_point": 10,
    "clip_trajs": False,
    "scale_types": ["step_std", "mean_time_step"],
    "log_features": True,
    "data_type": "no_features",  # no features because features are all computed by the model
    "edge_method": "geom_causal",
}


def _get_encoder_structure():
    return tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(16)),
            tf.keras.layers.Dense(units=256),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Dense(units=16),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Dense(units=2),
        ]
    )


def train_model(
    export_path: str,
    time_delta_range: tuple,
    max_n_epochs: int = 100,
    num_workers: int = 0,  # number of workers during the training process (should be < # CPUs)
    dim: int = 2,  # Dimension of trajectories. 2 or 3
    RW_types: List = ["fBM",
            "LW",
            "sBM",
            "OU",
            "CTRW"],
    log_diffusion_range: tuple = (
        -2.0,
        1.1,
    ),  # log-diffusion is drawn following a truncated centered gaussian in this range
    length_range: tuple = (
        7,
        35,
    ),  # length is drawn from a shifted (+min) and truncated (<max) exponential whose scale is (max-min)/8
    noise_range: tuple = (
        0.015,
        0.05,
    ),  # localization uncertainty, in micrometers (one value per trajectory)
    predict_alpha: bool = True, # Whether the model should be trained to predict alpha or not
    predict_model: bool = True, # Whether the model should be trained to predict RW type or not
):

    if not os.path.exists(export_path):
        os.mkdir(export_path)
    assert os.path.isdir(export_path)

    dl_params = {"batch_size": 128, "num_workers": num_workers}

    ds_params = {
        "dim": dim,  # can be (1, 2 or 3)
        "RW_types": RW_types,  # Types of random walks used during training
        "time_delta_range": time_delta_range,
        "logdiffusion_range": log_diffusion_range,
        "length_range": length_range,
        "noise_range": noise_range,
        "N": int(1e5),
    }

    model = MainNet(
        # tasks=["alpha", "model"],
        n_c=32,
        latent_dim=16,
        lr=1e-3,
        dim=2,
        RW_types=ds_params["RW_types"],
        scale_types=["step_std", "mean_time_step"],
        predict_alpha = predict_alpha,
        predict_model = predict_model
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
        accelerator='auto', 
        devices=1,
        #gpus=1 * torch.cuda.is_available(),
        gradient_clip_val=10.0,
        reload_dataloaders_every_n_epochs=1,
        callbacks=[
            ES,
            LRM,
            CKPT,
            Plotter(),
            # LatentSpaceSaver(),
            # StochasticWeightAveraging(swa_epoch_start=10),
        ],
        log_every_n_steps=50,
        max_epochs=max_n_epochs,
        detect_anomaly=True,
        track_grad_norm=2,
        logger=tb_logger,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainer.fit(model=model, datamodule=dm)

    results = trainer.test(model, dm.test_dataloader())
    print(results)

    model.eval()
    output, target, h = get_predictions_of_dl(
        model, dm.test_dataloader(), latent_samples=int(3e4)
    )

    umap_params = {
        "min_dist": 0.5,
        "n_components": 2,
        "verbose": True,
        "n_neighbors": 200,
        "metric": "cosine",
    }

    encoder = _get_encoder_structure()

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

    return model, encoder


def load_model(export_path: str):
    model = MainNet.load_from_checkpoint(os.path.join(export_path, "model.ckpt"))
    model.eval()
    model.freeze()

    encoder = tf.keras.models.load_model(os.path.join(export_path, "umap"))
    return model, encoder


def plot_demo(
    model: MainNet,
    encoder: tf.keras.Model,
    num_workers: int = 0,
    dim: int = 2,
    time_delta: float = 0.03,
    log_diffusion_range: tuple = (-2.0, 1.1),
    length_range: tuple = (7, 35),
    noise_range: tuple = (0.015, 0.05),
):

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
        "N": int(1e5),
    }

    pl.seed_everything(1)
    dm = DataModule(ds_params=ds_params, dl_params=dl_params, graph_info=graph_info)
    dm.setup(plot=False)

    trainer = pl.Trainer(
        auto_select_gpus=torch.cuda.is_available(), gpus=1 * torch.cuda.is_available()
    )

    trainer.test(model, dm.test_dataloader())

    output, target, h = get_predictions_of_dl(
        model, dm.test_dataloader(), latent_samples=int(3e4)
    )

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


def trajectory_is_valid(t: pd.DataFrame):
    if not t.shape[0] >= 7:
        return False
    if not t["t"].is_monotonic_increasing:
        return False
    return True


def get_predictions(
    model,
    encoder,  # UMAP
    trajectories: Union[List, pd.DataFrame],
    times: List = None,  # must be provided if trajectories is a list
):
    assert len(trajectories) > 0, "Empty list of trajectories"

    if isinstance(trajectories, pd.DataFrame):
        trajectories_indices = [
            (n, t[["x", "y"]].values, t["t"].values)
            for n, t in trajectories.sort_values(["frame", "t"]).groupby("n")
            if trajectory_is_valid(t)
        ]
        indices = np.array([_[0] for _ in trajectories_indices])
        trajectories = [_[1] for _ in trajectories_indices]
        times = [_[2] for _ in trajectories_indices]
    else:
        assert (
            times is not None
        ), "if trajectories is a list, then the 'times' argument must be provided"
        assert len(times) == len(
            trajectories
        ), "times and trajectories must have the same length"
        indices = np.arange(len(trajectories))

    lengths = [t.shape[0] for t in trajectories]

    exp_ds = ExpTrajDataSet(
        dim=trajectories[0].shape[1],
        graph_info=graph_info,
        trajs=trajectories,
        times=times,
    )
    dl_exp = DataLoader(exp_ds, batch_size=128, shuffle=False, drop_last=False)

    outputs, info, h = get_predictions_of_dl(model, dl_exp, latent_samples=len(exp_ds))
    
    df = pd.DataFrame(index=indices)
    
    if "model" in outputs:
        probabilities = np.exp(outputs["model"]) / np.reshape(
        np.exp(outputs["model"]).sum(axis=1), (-1, 1)
    )
        best_model = [
            model.hparams["RW_types"][i] for i in np.argmax(probabilities, axis=1)
        ]
        df["best_model"] = best_model
        df[["p_%s" % m for m in model.hparams["RW_types"]]] = probabilities
        
    if "alpha" in outputs:
        df["alpha"] = outputs["alpha"]
        
    df["length"] = lengths
    df[["U_1", "U_2"]] = np.array(encoder(h))
    h_cols = ["h_%d" % (i + 1) for i in range(h.shape[1])]
    df[h_cols] = h

    return df
