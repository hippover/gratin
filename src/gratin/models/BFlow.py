from collections import defaultdict
import pytorch_lightning as pl
from pytorch_lightning.metrics.regression import MeanAbsoluteError as MAE
from pytorch_lightning.metrics.regression import MeanSquaredError as MSE
from pytorch_lightning.metrics import F1
from pytorch_lightning.metrics import ExplainedVariance as EV
import torch.nn as nn
import torch
from functools import partial
from ..layers.diverse import batch_from_positions, batch_from_sub_batches
from ..layers.features_init import *
from ..training.network_tools import L2_loss, Category_loss, is_concerned
from ..data.data_classes import DataModule
from ..layers.encoders import *
from ..layers.InvNet import InvertibleNet
from ..layers.fBMGenerator import fBMGenerator
from torch.optim.lr_scheduler import ExponentialLR


class BFlowFBM(pl.LightningModule):
    def __init__(
        self,
        n_c: int,  # number of convolutions
        latent_dim: int,
        dim: int,  # traj dim
        gamma: float = 0.98,
        lr: float = 1e-3,
        scale_types: list = ["step_std"],
        tau_range: tuple = (5, 10),
        alpha_range: tuple = (0.4, 1.6),
        T: int = 100,
        n_lengths: int = 8,
        degree: int = 10,
        mode: str = "alpha_tau",  # either "alpha_tau" or "alpha_diff"
        **kwargs
    ):
        super().__init__()

        self.save_hyperparameters()
        x_dim = TrajsFeatures.x_dim(scale_types)
        e_dim = TrajsFeatures.e_dim(scale_types)
        self.save_hyperparameters({"x_dim": x_dim, "e_dim": e_dim})

        if self.hparams["mode"] == "alpha_tau":
            self.T_values = [self.hparams["T"]]
        elif self.hparams["mode"] == "alpha_diff":
            self.T_values = np.logspace(
                1,
                np.log10(self.hparams["T"]),
                dtype=int,
                num=n_lengths,
            )
        else:
            raise NotImplementedError("Mode inconnu : %s" % mode)

        self.generator = fBMGenerator(dim=self.hparams["dim"])

        self.features_maker = TrajsFeatures()

        self.summary_net = TrajsEncoder2(
            n_c=n_c,
            latent_dim=latent_dim,
            x_dim=self.hparams["x_dim"],
            e_dim=self.hparams["e_dim"],
            traj_dim=0,  # On se fiche de l'orientation
            n_scales=len(scale_types) + 1
            if "diff" in self.hparams["mode"]
            else 0,  # + 1 because we add time as a scale
        )

        self.dim_theta = 2

        self.MAE_alpha = MAE()
        self.MSE_tau = MSE()
        self.MSE_diff = MSE()

        self.invertible_net = InvertibleNet(dim_theta=2, dim_x=latent_dim, n_blocks=1)

        self.norm_dist = torch.distributions.normal.Normal(0.0, 1, validate_args=None)

        print(self.hparams)

        self.check_eigenvalues()

    def check_eigenvalues(self):

        # On vérifie qu'aux extrémités de sintervalles, la matrice de corrélation est positive

        a_min = torch.ones((1),device="cuda")*self.hparams["alpha_range"][0]
        a_max = torch.ones((1),device="cuda")*self.hparams["alpha_range"][1]
        tau_min = torch.ones((1),device="cuda")*self.hparams["tau_range"][0]
        tau_max = torch.ones((1),device="cuda")*self.hparams["tau_range"][1]
        diffusion = torch.ones_like(tau_min)

        self.generator(a_min,tau_min,diffusion,T=self.hparams["T"])
        self.generator(a_max,tau_min,diffusion,T=self.hparams["T"])
        self.generator(a_min,tau_max,diffusion,T=self.hparams["T"])
        self.generator(a_max,tau_max,diffusion,T=self.hparams["T"])
        print("Checked that correlation matrices are pos-def")

    def scale(self, param, range, inverse):
        m, M = range
        if inverse == False:
            return self.norm_dist.icdf((param - m) / (M - m))
        elif inverse == True:
            return self.norm_dist.cdf(param) * (M - m) + m

    def scale_alpha(self, alpha, inverse=False):
        return self.scale(alpha, self.hparams["alpha_range"], inverse)

    def scale_logtau(self, logtau, inverse=False):
        return self.scale(
            logtau,
            (
                np.log10(self.hparams["tau_range"][0]),
                np.log10(self.hparams["tau_range"][1]),
            ),
            inverse,
        )

    def scale_logdiff(self, logdiff, inverse=False):
        return self.scale(logdiff, (-2, 2), inverse)

    def make_theta(self, x):
        if self.hparams["mode"] == "alpha_tau":
            return torch.cat(
                (self.scale_alpha(x.alpha), self.scale_logtau(x.log_tau)), dim=1
            ).float()
        elif self.hparams["mode"] == "alpha_diff":
            return torch.cat(
                (self.scale_alpha(x.alpha), self.scale_logdiff(x.log_diffusion)), dim=1
            ).float()
        else:
            raise NotImplementedError("Unknown mode %s" % self.hparams["mode"])

    def get_params(self, theta):
        if self.hparams["mode"] == "alpha_tau":
            return {
                "alpha": self.scale_alpha(theta[:, 0].view(-1, 1), inverse=True),
                "log_tau": self.scale_logtau(theta[:, 1].view(-1, 1), inverse=True),
            }
        elif self.hparams["mode"] == "alpha_diff":
            return {
                "alpha": self.scale_alpha(theta[:, 0].view(-1, 1), inverse=True),
                "log_diffusion": self.scale_logdiff(
                    theta[:, 1].view(-1, 1), inverse=True
                ),
            }
        else:
            raise NotImplementedError("Unknown mode %s" % self.hparams["mode"])

    def generate_batch_like(self, x):
        batches = []
        BS = torch.max(x.batch) + 1
        SBS = BS // len(self.T_values)
        for T in self.T_values:
            # ALPHA
            alpha = (
                torch.rand(SBS, device="cuda")
                * (self.hparams["alpha_range"][1] - self.hparams["alpha_range"][0])
                + self.hparams["alpha_range"][0]
            )

            # TAU if needed
            if self.hparams["mode"] == "alpha_tau":
                log_tau = torch.rand(SBS, device="cuda") * (
                    np.log10(self.hparams["tau_range"][1])
                    - np.log10(self.hparams["tau_range"][0])
                ) + np.log10(self.hparams["tau_range"][0])
            elif self.hparams["mode"] == "alpha_diff":
                log_tau = torch.ones_like(alpha) * np.log10(T)
            tau = torch.pow(10.0, log_tau)
            # Make sure that tau is not larger than T
            # tau = torch.where(tau > T, T, tau)

            # DIFFUSION
            log_diffusion = torch.rand(SBS, device="cuda") * 4 - 2
            diffusion = torch.pow(10.0, log_diffusion)

            pos = self.generator(alpha, tau, diffusion, T)
            x = batch_from_positions(
                pos,
                N=SBS,
                L=T,
                D=self.hparams["dim"],
                degree=self.hparams["degree"],
            )
            x.alpha = alpha.view(-1, 1)
            x.log_tau = log_tau.view(-1, 1)
            x.log_diffusion = log_diffusion.view(-1, 1)
            batches.append(x)

        x = batch_from_sub_batches(batches)
        return x

    def forward(self, x, sample=False, n_repeats=1, batch_idx=0):

        # Si x ne contient pas de trajectoires, on en génère
        if torch.isnan(x.pos).sum() > 0:
            x = self.generate_batch_like(x)
            assert torch.isnan(x.pos).sum() == 0

        # NORMAL FORWARD PASS

        X, E, scales, orientation = self.features_maker(
            x, scale_types=self.hparams["scale_types"]
        )
        x.adj_t = x.adj_t.set_value(E)
        x.x = X
        if "diff" in self.hparams["mode"]:
            x.scales = scales
        # print("x.scales")
        # print(scales)
        # x.orientation = orientation

        assert x.x.shape[1] == self.hparams["x_dim"]
        h = self.summary_net(x)

        true_theta = self.make_theta(x)

        if not sample:
            z, log_J = self.invertible_net(true_theta, h, inverse=False)
            return z, log_J

        elif sample:
            z = torch.normal(
                mean=0.0,
                std=1.0,
                size=(n_repeats * h.shape[0], self.dim_theta),
                device=h.device,
            )
            # print(x.shape)
            h = h.repeat_interleave(n_repeats, 0)

            theta = self.invertible_net(z, h, inverse=True)
            return theta, true_theta

    def training_step(self, batch, batch_idx):
        z, log_J = self(batch, batch_idx=batch_idx)
        z_norm2 = torch.sum(z ** 2, dim=1)
        l = torch.mean(0.5 * z_norm2 - log_J, dim=0)
        self.log("training_loss", value=l, on_step=False, on_epoch=True)
        self.log("z_norm", value=torch.mean(z_norm2), on_step=True)
        self.log("log_J", value=torch.mean(log_J), on_step=True)

        return {"loss": l}

    def sample_step(self, batch, batch_idx):
        theta, true_theta = self(batch, batch_idx=batch_idx, sample=True)
        if self.hparams["mode"] == "alpha_tau":
            self.MAE_alpha(
                self.scale_alpha(theta[:, 0], inverse=True),
                self.scale_alpha(true_theta[:, 0], inverse=True),
            )
            self.MSE_tau(
                self.scale_logtau(theta[:, 1], inverse=True),
                self.scale_logtau(true_theta[:, 1], inverse=True),
            )
        elif self.hparams["mode"] == "alpha_diff":
            self.MAE_alpha(
                self.scale_alpha(theta[:, 0], inverse=True),
                self.scale_alpha(true_theta[:, 0], inverse=True),
            )
            self.MSE_diff(
                self.scale_logdiff(theta[:, 1], inverse=True),
                self.scale_logdiff(true_theta[:, 1], inverse=True),
            )
        preds = self.get_params(theta)
        targets = self.get_params(true_theta)
        return preds, targets

    def log_metrics(self, step="test"):
        if self.hparams["mode"] == "alpha_tau":
            self.log(
                "MAE_alpha_%s" % step, self.MAE_alpha, on_step=False, on_epoch=True
            )
            self.log("MSE_tau_%s" % step, self.MSE_tau, on_step=False, on_epoch=True)
        elif self.hparams["mode"] == "alpha_diff":
            self.log(
                "MAE_alpha_%s" % step, self.MAE_alpha, on_step=False, on_epoch=True
            )
            self.log("MSE_diff_%s" % step, self.MSE_diff, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        preds, targets = self.sample_step(batch, batch_idx=batch_idx)
        self.log_metrics(step="test")
        return {"targets": targets, "preds": preds}

    def on_test_epoch_end(self):
        self.logger.log_hyperparams(self.hparams)

    def validation_step(self, batch, batch_idx):
        preds, targets = self.sample_step(batch, batch_idx=batch_idx)
        self.log_metrics(step="val")
        return {"targets": targets, "preds": preds}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), amsgrad=True, lr=self.hparams["lr"]
        )
        scheduler = ExponentialLR(optimizer, gamma=self.hparams["gamma"])
        return [optimizer], [scheduler]
