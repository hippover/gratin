from collections import defaultdict
import pytorch_lightning as pl
from torchmetrics import MeanAbsoluteError as MAE
from torchmetrics.classification.f_beta import F1Score as F1
import torch.nn as nn
import torch
from functools import partial
from ..layers.diverse import MLP, AlphaPredictor
from ..layers.features_init import *
from ..training.network_tools import L2_loss, Category_loss, is_concerned
from ..layers.encoders import *
from torch.optim.lr_scheduler import ExponentialLR
import scipy
import logging
from ..layers.mmd_pytorch import MMD_loss


class MMDNet(pl.LightningModule):
    def __init__(
        self,
        n_c: int,  # number of convolutions
        latent_dim: int,
        dim: int,  # traj dim
        gamma: float = 0.98,
        lr: float = 1e-3,
        scale_types: list = ["step_std", "mean_time_step"],
    ):
        super().__init__()

        if "mean_time_step" not in scale_types:
            logging.warn(
                'The network has no information about time steps. To add some, use "mean_time_step" as a scale'
            )

        self.save_hyperparameters()
        x_dim = TrajsFeaturesSimple.x_dim()
        e_dim = TrajsFeaturesSimple.e_dim()
        self.save_hyperparameters({"x_dim": x_dim, "e_dim": e_dim})

        self.encoder = TrajsEncoder(
            traj_dim=1,
            x_dim=TrajsFeaturesSimple.x_dim(),
            e_dim=TrajsFeaturesSimple.e_dim(),
            n_c=n_c,
            latent_dim=latent_dim,
            n_scales=len(scale_types),
        )

        self.mmd_layer = MMD_loss(kernel_type="rbf", kernel_num=1, fix_sigma=1.0)
        self.features_maker = TrajsFeaturesSimple()

    def forward(self, x):

        X, E, scales = self.features_maker(
            x,
        )
        x.adj_t = x.adj_t.set_value(E, layout="coo")
        x.x = X
        x.scales = torch.cat(
            [scales[k].view(-1, 1) for k in self.hparams["scale_types"]], dim=1
        )

        assert x.x.shape[1] == self.hparams["x_dim"]
        h = self.encoder(x)
        return h

    def training_step(self, batch, stage="train"):
        # Part of the computation that is common to train, val and test
        h = self(batch)
        n = h.shape[0] // 2
        h1, h2 = h[:n], h[n:]
        return self.mmd_layer(h1, h2)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), amsgrad=True, lr=self.hparams["lr"], maximize=True
        )
        scheduler = ExponentialLR(optimizer, gamma=self.hparams["gamma"])
        return [optimizer], [scheduler]
