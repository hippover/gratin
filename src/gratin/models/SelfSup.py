from collections import defaultdict
import pytorch_lightning as pl
from pytorch_lightning.metrics.regression import MeanAbsoluteError as MAE
from pytorch_lightning.metrics.regression import MeanSquaredError as MSE
from pytorch_lightning.metrics import F1
from pytorch_lightning.metrics import ExplainedVariance as EV
import torch.nn as nn
import torch
from functools import partial
from ..layers.diverse import MLP, AlphaPredictor, batch_from_positions
from ..layers.features_init import *
from ..training.network_tools import L2_loss, Category_loss, is_concerned
from ..data.data_classes import DataModule

from ..layers.encoders import *
from ..layers.InvNet import InvertibleNet

from torch.optim.lr_scheduler import ExponentialLR
from torch_geometric.data import Batch


class SelfSup(pl.LightningModule):
    def __init__(
        self,
        n_c: int,  # number of convolution channels
        latent_dim: int,
        dim: int,  # traj dim
        gamma: float = 0.98,
        lr: float = 1e-3,
        scale_types: list = ["step_std"],
        degree: int = 20,
        L_pred: int = 10,
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

        self.steps_pred = InvertibleNet(
            (L_pred - 1) * dim, latent_dim, n_blocks=3, stable_s=True
        )

    def forward(self, x):

        X, E, scales, orientation = self.features_maker(
            x, scale_types=self.hparams["scale_types"]
        )
        x.adj_t = x.adj_t.set_value(E)
        x.x = X
        # x.scales = scales
        # x.orientation = orientation

        assert x.x.shape[1] == self.hparams["x_dim"]
        h_true = self.summary_net(x)

        L = self.hparams["L_pred"]
        D = self.hparams["dim"]
        N = h_true.shape[0]

        # z = torch.randn((N, (L - 1) * D), device=self.device)
        # raw_steps = self.steps_pred(torch.cat((h_true), dim=1))
        epsilon = torch.randn((N, (L - 1) * D), device=h_true.device)
        raw_steps, log_J = self.steps_pred(epsilon, h_true)
        steps = raw_steps.view((N, D, L - 1))

        # assert torch.equal(raw_steps[0, (L - 1) * D - 1], steps[0, D - 1, L - 2])
        # assert torch.equal(raw_steps[1, (L - 1) * D - 1], steps[1, D - 1, L - 2])
        # if D >= 2:
        #    assert torch.equal(
        #        raw_steps[1, (L - 1) * (D - 1) - 1], steps[1, D - 2, L - 2]
        #    )
        pos = torch.cat(
            (
                torch.zeros((N, D, 1), device=raw_steps.device),
                torch.cumsum(steps, dim=2),
            ),
            dim=2,
        )
        pos = torch.transpose(pos, 1, 2)

        x_pred = batch_from_positions(pos, N=N, L=L, D=D, degree=self.hparams["degree"])

        (X_pred, E_pred, scales, orientation) = self.features_maker(
            x_pred,
            scale_types=self.hparams["scale_types"],
            return_intermediate=False,
        )
        # Need to re-create a batch to set the edge features values...
        x_pred = Batch(
            x_pred.batch, adj_t=x_pred.adj_t.set_value(E_pred), pos=x_pos, x=X_pred
        )

        h_pred = self.summary_net(x_pred)

        return h_true, h_pred, x_pos, log_J

    def training_step(self, batch, batch_idx):
        h_true, h_pred, traj_pred, log_J = self(batch)
        sigma_2 = 0.5 * torch.min(torch.var(h_true, dim=0) + torch.var(h_pred, dim=0))
        log_distance = torch.log(
            1 - torch.mean(nn.functional.cosine_similarity(h_true, h_pred))
        )
        # log_distance = torch.mean(torch.log(1e-5 + (h_true - h_pred) ** 2))
        h_norm = torch.mean(h_true ** 2 + h_pred ** 2)
        l = (
            log_distance
            + h_norm
            + (1.0 / (1e-5 + sigma_2))
            - 1.0 / torch.sqrt(sigma_2 + 1e-5)
            # + torch.mean(log_J ** 2)
        )

        debug = False
        if debug:
            for n, p in self.named_parameters():
                if p.grad is None:
                    print(n, p.grad)
                elif torch.sum(torch.isnan(p.grad)) == 0:
                    print(n, "numeric")
                else:
                    print(n, "============  NULL")

        self.log("training_loss", value=l, on_epoch=True, prog_bar=True)
        self.log("log_distance", value=log_distance, prog_bar=True)
        self.log("inv_sigma_2", value=1.0 / sigma_2, prog_bar=True)
        self.log("h_norm", value=h_norm, prog_bar=True)
        # self.log("log_J", value=torch.mean(log_J), prog_bar=True) # 0 if stable_s

        return {"loss": l}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), amsgrad=True, lr=self.hparams["lr"]
        )
        scheduler = ExponentialLR(optimizer, gamma=self.hparams["gamma"])
        return [optimizer], [scheduler]

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu,
        using_native_amp,
        using_lbfgs,
    ):
        for n, p in self.named_parameters():
            if p.grad is None:
                continue
            if torch.sum(torch.isnan(p.grad)) > 0:
                print("%s has a nan gradient" % n)
                return
        optimizer.step(closure=optimizer_closure)
