from collections import defaultdict
import pytorch_lightning as pl
from pytorch_lightning.metrics.regression import MeanAbsoluteError as MAE
from pytorch_lightning.metrics.regression import MeanSquaredError as MSE
from pytorch_lightning.metrics import F1
from pytorch_lightning.metrics import ExplainedVariance as EV
import torch.nn as nn
import torch
from functools import partial
from ..layers.diverse import batch_from_positions
from ..layers.features_init import *
from ..training.network_tools import L2_loss, Category_loss, is_concerned
from ..data.data_classes import DataModule
from ..layers.encoders import *
from ..layers.InvNet import InvertibleNet
from ..layers.fBMGenerator import fBMGenerator
from torch.optim.lr_scheduler import ExponentialLR


class BFlow(pl.LightningModule):
    def __init__(
        self,
        n_c: int,  # number of convolutions
        latent_dim: int,
        dim: int,  # traj dim
        gamma: float = 0.98,
        lr: float = 1e-3,
        scale_types: list = ["step_std"],
        **kwargs
    ):
        super().__init__()

        self.save_hyperparameters()
        x_dim = TrajsFeatures.x_dim(scale_types)
        e_dim = TrajsFeatures.e_dim(scale_types)
        self.save_hyperparameters({"x_dim": x_dim, "e_dim": e_dim})

        self.features_maker = TrajsFeatures()

        self.summary_net = TrajsEncoder2(
            n_c=n_c,
            latent_dim=latent_dim,
            x_dim=self.hparams["x_dim"],
            e_dim=self.hparams["e_dim"],
            traj_dim=0,  # On se fiche de l'orientation
            n_scales=0,  # On se fiche de la diffusivité
        )

        self.dim_theta = 2

        self.MAE_alpha = MAE()
        self.MSE_tau = MSE()

        self.invertible_net = InvertibleNet(dim_theta=2, dim_x=latent_dim, n_blocks=1)

    def forward(self, x, sample=False, n_repeats=1):

        X, E, scales, orientation = self.features_maker(
            x, scale_types=self.hparams["scale_types"]
        )
        x.adj_t = x.adj_t.set_value(E)
        x.x = X
        # x.scales = scales
        # x.orientation = orientation

        assert x.x.shape[1] == self.hparams["x_dim"]
        h = self.summary_net(x)

        true_theta = torch.cat((x.alpha, x.log_tau), dim=1).float()

        if not sample:

            z, log_J = self.invertible_net(true_theta, h, inverse=False)
            return z, log_J
        if sample:
            z = torch.normal(
                mean=0.0,
                std=1.0,
                size=(n_repeats * h.shape[0], self.dim_theta),
                device=h.device,
            )
            # print(x.shape)
            h = h.repeat_interleave(n_repeats, 0)
            # print(x.shape)
            # print(z.shape)
            theta = self.invertible_net(z, h, inverse=True)
            return theta, true_theta

    def training_step(self, batch, batch_idx):
        z, log_J = self(batch)
        z_norm2 = torch.sum(z ** 2, dim=1)
        l = torch.mean(0.5 * z_norm2 - log_J, dim=0)
        self.log("training_loss", value=l, on_step=False, on_epoch=True)
        self.log("z_norm", value=torch.mean(z_norm2), on_step=True)
        self.log("log_J", value=torch.mean(log_J), on_step=True)

        return {"loss": l}

    def test_step(self, batch, batch_idx):
        theta = self(batch, sample=True)
        self.MAE_alpha(theta[:, 0], batch.alpha[:, 0])
        self.MSE_tau(theta[:, 1], batch.log_tau[:, 0])
        self.log("MAE_alpha_test", self.MAE_alpha, on_step=False, on_epoch=True)
        self.log("MSE_tau_test", self.MSE_tau, on_step=False, on_epoch=True)
        targets = {"alpha": batch.alpha, "log_tau": batch.log_tau}
        preds = {"alpha": theta[:, 0].view(-1, 1), "log_tau": theta[:, 1].view(-1, 1)}
        return {"targets": targets, "preds": preds}

    def on_test_epoch_end(self):
        self.logger.log_hyperparams(self.hparams)

    def validation_step(self, batch, batch_idx):
        theta, true_theta = self(batch, sample=True)
        self.MAE_alpha(theta[:, 0], true_theta[:, 0])
        self.MSE_tau(theta[:, 1], true_theta[:, 1])
        self.log("MAE_alpha_val", self.MAE_alpha, on_step=False, on_epoch=True)
        self.log("MSE_tau_val", self.MSE_tau, on_step=False, on_epoch=True)
        preds = {"alpha": theta[:, 0].view(-1, 1), "log_tau": theta[:, 1].view(-1, 1)}
        targets = {
            "alpha": true_theta[:, 0].view(-1, 1),
            "log_tau": true_theta[:, 1].view(-1, 1),
        }
        return {"targets": targets, "preds": preds}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), amsgrad=True, lr=self.hparams["lr"]
        )
        scheduler = ExponentialLR(optimizer, gamma=self.hparams["gamma"])
        return [optimizer], [scheduler]


class BFlowFBM(BFlow):
    def __init__(self, tau_range, alpha_range, T, degree, **kwargs):

        super(BFlowFBM, self).__init__(
            **kwargs,
            tau_range=tau_range,
            alpha_range=alpha_range,
            T=T,
            degree=degree,
        )
        self.generator = fBMGenerator(T=self.hparams["T"], dim=self.hparams["dim"])

    def forward(self, x, sample=False, n_repeats=1):

        # Si x ne contient pas de trajectoires, on en génère
        if torch.isnan(x.pos).sum() > 0:
            BS = torch.max(x.batch) + 1
            log_tau = torch.rand(BS, device="cuda") * (
                np.log10(self.hparams["tau_range"][1])
                - np.log10(self.hparams["tau_range"][0])
            ) + np.log10(self.hparams["tau_range"][0])
            tau = torch.pow(10.0, log_tau)
            alpha = (
                torch.rand(BS, device="cuda")
                * (self.hparams["alpha_range"][1] - self.hparams["alpha_range"][0])
                + self.hparams["alpha_range"][0]
            )
            pos = self.generator(alpha, tau)
            x = batch_from_positions(
                pos,
                N=BS,
                L=self.generator.T,
                D=self.hparams["dim"],
                degree=self.hparams["degree"],
            )
            x.alpha = alpha.view(-1, 1)
            x.log_tau = log_tau.view(-1, 1)

        return super().forward(x, sample=sample, n_repeats=n_repeats)