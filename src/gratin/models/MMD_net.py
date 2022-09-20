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
from typing import List

from torch import pca_lowrank


def pseudo_alpha(alpha):
    # For batch.alpha when dealing with OU
    alpha_ = alpha.clone()
    alpha_[torch.isnan(alpha_)] = 0.5
    return alpha_


class MMDNet(pl.LightningModule):
    def __init__(
        self,
        n_c: int,  # number of convolutions
        latent_dim: int,
        dim: int,  # traj dim
        RW_types: List = ["sBM", "fBM"],
        gamma: float = 0.98,
        lr: float = 1e-3,
        noise_scale: float = 0.5,
        noise_scale_decay: float = 0.98,
        scale_types: list = ["step_std", "mean_time_step"],
        n_splits: int = 10,
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

        self.out_alpha = MLP(
            [latent_dim, 64, 64, 1], use_batch_norm=True, use_weight_norm=False
        )
        self.out_model = MLP(
            [latent_dim, 64, 64, len(RW_types)],
            use_batch_norm=True,
            use_weight_norm=False,
        )

        self.mmd_layer = MMD_loss(kernel_type="rbf", kernel_num=1, fix_sigma=1.0)
        self.features_maker = TrajsFeaturesSimple()
        self.cond_generators = {
            "alpha": self.gen_cond_alpha,
            "model": self.gen_cond_model,
            "diff": self.gen_cond_diff,
        }

    def gen_cond_alpha(self, batch):
        return (
            torch.abs(
                pseudo_alpha(batch.alpha[:, 0])
                - (torch.rand(1, device=batch.alpha.device) * 2)
            )
            < 0.3,
        )

    def gen_cond_model(self, batch):
        return (
            torch.isin(
                batch.model[:, 0],
                torch.randperm(
                    len(self.hparams["RW_types"]), device=batch.model.device
                )[: (len(self.hparams["RW_types"]) // 2)],
            ),
            pseudo_alpha(batch.alpha[:, 0]) > 1,
        )

    def gen_cond_diff(self, batch):
        return (
            torch.abs(
                batch.log_diffusion[:, 0]
                - (torch.rand(1, device=batch.log_diffusion.device) * 2.5 - 2.0)
            )
            < 0.5,
        )

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
        h_ = h.detach()
        alpha = self.out_alpha(h_)
        model = self.out_model(h_)
        return {"alpha": alpha, "model": model}, h

    def get_MMD(
        self, h_, batch, noise_amplitude: float = 0.0, return_s_penalty: bool = False
    ):
        u, s, v = pca_lowrank(h_, center=True)
        h = torch.matmul(h_, v @ torch.diag(1.0 / torch.sqrt(s + 1e-6)))
        s_penalty = torch.log(torch.min(s) / torch.max(s)) ** 2
        MIN_SPLIT_SIZE = h.shape[0] // 10

        mmds = {
            "alpha": 0.0,
            "model": 0.0,
            "diff": 0.0,
        }

        # Based on anomalous exponent :
        # those close to a value of alpha VS the others

        for i in range(self.hparams["n_splits"]):
            groups = {}
            for name, gen in self.cond_generators.items():
                while True:
                    # conds[name] is a list of boolean tensors
                    conds = gen(batch)
                    n_levels = len(conds)

                    g = torch.zeros_like(h[:, 0])
                    for i, cond in enumerate(conds[::-1]):
                        n = (n_levels - 1) - i
                        g = g + (2**n) * cond

                    if all(
                        [
                            torch.sum(g == i) > MIN_SPLIT_SIZE
                            for i in range(2**n_levels)
                        ]
                    ):
                        groups[name] = g
                        break

                for i in range(n_levels):
                    cond1 = g == (2 * i)
                    cond2 = g == (2 * i + 1)
                    h1, h2 = h[cond1, :], h[cond2, :]
                    h1 = h1 + torch.randn_like(h1) * noise_amplitude
                    h2 = h2 + torch.randn_like(h2) * noise_amplitude
                    mmds[name] = mmds[name] + (
                        1.0 / (2 ** (n_levels - 1))
                    ) * self.mmd_layer(h1, h2)

        mmd_alpha = mmds["alpha"] / self.hparams["n_splits"]
        mmd_model = mmds["model"] / self.hparams["n_splits"]
        mmd_diff = mmds["diff"] / self.hparams["n_splits"]

        to_return = (
            mmd_alpha,
            mmd_model,
            mmd_diff,
        )
        if return_s_penalty:
            to_return = to_return + (s_penalty,)
        return to_return

    def training_step(self, batch, batch_idx):
        out, h_ = self(batch)
        alpha = out["alpha"]
        model = out["model"]

        mmd_alpha, mmd_model, mmd_diff, s_penalty = self.get_MMD(
            h_,
            batch,
            noise_amplitude=self.hparams["noise_scale"]
            * (self.hparams["noise_scale_decay"] ** self.current_epoch),
            return_s_penalty=True,
        )

        total_mmd = mmd_model + mmd_alpha + mmd_diff

        alpha_valid = ~torch.isnan(batch.alpha)
        alpha_loss = torch.mean((alpha - batch.alpha)[alpha_valid] ** 2)
        model_loss = torch.mean(
            torch.nn.functional.cross_entropy(model, batch.model[:, 0])
        )
        # print(alpha_loss, model_loss)

        self.log(
            "MMD",
            {
                "alpha": mmd_alpha,
                "model": mmd_model,
                "diff": mmd_diff,
                "total": total_mmd,
            },
            on_epoch=True,
            on_step=True,
            batch_size=h_.shape[0],
        )

        self.log("alpha", alpha_loss)
        self.log("model", model_loss)
        self.log("log_lambda", s_penalty)

        return {
            "loss": (total_mmd - s_penalty) - (alpha_loss + model_loss),
            # "loss": mmd_model - s_penalty,
            "h": h_,
            "alpha": alpha,
            "model": model,
        }

    def validation_step(self, batch, batch_idx):
        _, h_ = self(batch)

        mmd_alpha, mmd_model, mmd_diff = self.get_MMD(
            h_,
            batch,
            noise_amplitude=0.0,
            return_s_penalty=False,
        )
        total_mmd = mmd_model + mmd_alpha + mmd_diff
        self.log(
            "MMD_val",
            {
                "alpha": mmd_alpha,
                "model": mmd_model,
                "diff": mmd_diff,
                "total": total_mmd,
            },
            on_epoch=True,
            on_step=True,
            batch_size=h_.shape[0],
        )
        self.log("main_MMD", total_mmd, on_epoch=True, batch_size=h_.shape[0])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), amsgrad=True, lr=self.hparams["lr"], maximize=True
        )
        scheduler = ExponentialLR(optimizer, gamma=self.hparams["gamma"])
        return [optimizer], [scheduler]
