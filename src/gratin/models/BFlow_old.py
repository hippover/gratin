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
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR


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
            # alternative 1 : sample T in log
            # self.T_values = np.logspace(
            #    1,
            #    np.log10(self.hparams["T"]),
            #    dtype=int,
            #    num=n_lengths,
            # )
            # altetnative 2 : samlpe T uniformly
            self.T_values = np.linspace(10, self.hparams["T"], dtype=int, num=n_lengths)
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

        self.invertible_net = InvertibleNet(dim_theta=2, dim_x=latent_dim, n_blocks=3)

        self.norm_dist = torch.distributions.normal.Normal(0.0, 1, validate_args=None)

        self.alpha_range = self.hparams["alpha_range"]
        self.tau_range = self.hparams["tau_range"]
        if self.hparams["mode"] != "alpha_tau":
            self.tau_range = (self.hparams["T"], self.hparams["T"] + 1)
        self.check_eigenvalues()

        print(self.hparams)

    def check_eigenvalues(self):

        # On vérifie qu'aux extrémités des intervalles, la matrice de corrélation est positive
        OK = False
        a_range_init = self.alpha_range
        while not OK:
            a_min = torch.ones((1), device="cuda") * self.alpha_range[0]
            a_max = torch.ones((1), device="cuda") * self.alpha_range[1]

            tau_min = torch.ones((1), device="cuda") * self.tau_range[0]
            tau_max = torch.ones((1), device="cuda") * self.tau_range[1]
            diffusion = torch.ones_like(tau_min)

            try:
                for T in self.T_values:
                    self.generator(a_min, tau_min, diffusion, T=T)
                    self.generator(a_min, tau_max, diffusion, T=T)
            except Exception as e:
                self.alpha_range = (self.alpha_range[0] + 0.05, self.alpha_range[1])
                print(e)
                continue
            try:
                for T in self.T_values:
                    self.generator(a_max, tau_min, diffusion, T=T)
                    self.generator(a_max, tau_max, diffusion, T=T)
            except Exception as e:
                self.alpha_range = (self.alpha_range[0], self.alpha_range[1] - 0.05)
                print(e)
                continue
            OK = True
        a_range_end = self.alpha_range
        print("Checked that correlation matrices are pos-def")
        print("alpha range changed from")
        print(a_range_init)
        print("to")
        print(a_range_end)

    def scale(self, param, range, inverse):
        # bypass, no scaling
        # return param
        m, M = range
        # To avoid infinity, we slightly widen the range
        range_size = M - m
        m -= 0.05 * range_size
        M += 0.05 * range_size
        if inverse == False:
            return self.norm_dist.icdf((param - m) / (M - m))
        elif inverse == True:
            return self.norm_dist.cdf(param) * (M - m) + m

    def scale_alpha(self, alpha, inverse=False):
        return self.scale(alpha, self.alpha_range, inverse)

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
        # print("BS = %d" % BS)
        SBS_min = BS // len(self.T_values)
        # print("SBS = %s" % SBS)
        for T in self.T_values:
            # On fait plus de trajectoires courtes que de longues,
            # pour que chaque layer ait vu autant de messages venant de trajectoires courtes que de trajectoires longues
            # SBS = int(SBS_min * np.max(self.T_values) / T)
            SBS = int(SBS_min)
            # print("T = %d, generating %d trajs" % (T, SBS))
            # ALPHA
            alpha = (
                torch.rand(SBS, device="cuda")
                * (self.alpha_range[1] - self.alpha_range[0])
                + self.alpha_range[0]
            )

            # TAU if needed
            if self.hparams["mode"] == "alpha_tau":
                log_tau = torch.rand(SBS, device="cuda") * (
                    np.log10(self.hparams["tau_range"][1])
                    - np.log10(self.hparams["tau_range"][0])
                ) + np.log10(self.hparams["tau_range"][0])
            elif self.hparams["mode"] == "alpha_diff":
                log_tau = torch.ones_like(alpha) * np.log10(T) + 1
            tau = torch.pow(10.0, log_tau)
            # Make sure that tau is not larger than T
            # tau = torch.where(tau > T, T, tau)

            # DIFFUSION
            log_diffusion = torch.rand(SBS, device="cuda") * 4 - 2
            diffusion = torch.pow(10.0, log_diffusion)

            pos = self.generator(alpha, tau, diffusion, T)
            # print("Generating T = %d" % T)
            # print(pos.shape)

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
            x.length = torch.ones_like(x.alpha) * T
            assert x.batch.shape[0] == SBS * T
            batches.append(x)

        # print("num of sub_batches %d " % len(batches))

        x = batch_from_sub_batches(batches)

        return x

    def forward(self, x, sample=False, n_repeats=1, batch_idx=0, return_input=False):

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
            if not return_input:
                return theta, true_theta
            else:
                return theta, true_theta, x

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        z, log_J = self(batch, batch_idx=batch_idx)
        z_norm2 = torch.sum(z**2, dim=1)
        l = torch.mean(0.5 * z_norm2 - log_J, dim=0)
        self.log("training_loss", value=l, on_step=False, on_epoch=True, prog_bar=True)
        self.log("z_norm", value=torch.mean(z_norm2), on_step=True, prog_bar=True)
        self.log("log_J", value=torch.mean(log_J), on_step=True, prog_bar=True)

        return {"loss": l}

    def sample_step(self, batch, batch_idx):
        n_repeats = 20
        theta, true_theta = self(
            batch, batch_idx=batch_idx, sample=True, n_repeats=n_repeats
        )
        true_theta = true_theta.repeat_interleave(n_repeats, 0)
        preds = self.get_params(theta)
        targets = self.get_params(true_theta)

        self.MAE_alpha(preds["alpha"], targets["alpha"])
        if self.hparams["mode"] == "alpha_tau":
            self.MSE_tau(preds["log_tau"], targets["log_tau"])
        elif self.hparams["mode"] == "alpha_diff":
            self.MSE_diff(
                preds["log_diffusion"],
                targets["log_diffusion"],
            )

        return preds, targets

    def log_metrics(self, step="test"):
        if self.hparams["mode"] == "alpha_tau":
            self.log(
                "hp/MAE_alpha_%s" % step,
                self.MAE_alpha,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                "hp/MSE_tau_%s" % step,
                self.MSE_tau,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        elif self.hparams["mode"] == "alpha_diff":
            self.log(
                "hp/MAE_alpha_%s" % step,
                self.MAE_alpha,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                "hp/MSE_diff_%s" % step,
                self.MSE_diff,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

    def test_step(self, batch, batch_idx):
        preds, targets = self.sample_step(batch, batch_idx=batch_idx)
        self.log_metrics(step="test")
        return {"targets": targets, "preds": preds}

    def on_train_start(self):
        self.logger.log_hyperparams(
            self.hparams,
            {
                "hp/MSE_diff_val": np.inf,
                "hp/MSE_diff_test": np.inf,
                "hp/MAE_alpha_val": np.inf,
                "hp/MAE_alpha_test": np.inf,
            },
        )

    def on_validation_epoch_end(self):
        print("Validation epoch end")
        self.logger.experiment.flush()

    def validation_step(self, batch, batch_idx):
        preds, targets = self.sample_step(batch, batch_idx=batch_idx)
        self.log_metrics(step="val")
        return {"targets": targets, "preds": preds}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [
                {"params": self.summary_net.parameters(), "lr": 1e-4},
                {
                    "params": self.invertible_net.parameters(),
                    "lr": self.hparams["lr"],
                },
            ],
            amsgrad=True,
        )

        graph_lambda = (
            lambda step: (self.hparams["gamma"] ** int(step // 150))
            if step >= 1500
            else 0.0
        )
        inv_lambda = (
            lambda step: (self.hparams["gamma"] ** int(step // 150))
            # if step >= 1500
            # else 0.0
        )

        scheduler = {
            "scheduler": LambdaLR(optimizer, [graph_lambda, inv_lambda]),
            "interval": "step",
        }
        return [optimizer], [scheduler]
