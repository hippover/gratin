from collections import defaultdict
import pytorch_lightning as pl

try:

    from pytorch_lightning.metrics.regression import MeanAbsoluteError as MAE
    from pytorch_lightning.metrics import F1
    from pytorch_lightning.metrics import ExplainedVariance as EV
except:
    from torchmetrics import MeanAbsoluteError as MAE
    from torchmetrics.classification.f_beta import F1Score as F1
    from torchmetrics import ExplainedVariance as EV
import torch.nn as nn
import torch
from functools import partial
from ..layers.diverse import MLP, AlphaPredictor
from ..layers.features_init import *
from ..training.network_tools import L2_loss, Category_loss, is_concerned
from ..layers.encoders import *
from torch.optim.lr_scheduler import ExponentialLR


class MainNet(pl.LightningModule):
    def __init__(
        self,
        tasks: list,
        n_c: int,  # number of convolutions
        latent_dim: int,
        dim: int,  # traj dim
        gamma: float = 0.98,
        lr: float = 1e-3,
        RW_types: list = ["fBM"],
        scale_types: list = ["step_std"],
    ):
        super().__init__()

        self.save_hyperparameters()
        x_dim = TrajsFeatures.x_dim(scale_types)
        e_dim = TrajsFeatures.e_dim(scale_types)
        self.save_hyperparameters({"x_dim": x_dim, "e_dim": e_dim})

        self.encoder = TrajsEncoder2(
            n_c=n_c,
            latent_dim=latent_dim,
            x_dim=self.hparams["x_dim"],
            e_dim=self.hparams["e_dim"],
            traj_dim=dim,
            n_scales=len(self.hparams["scale_types"])
            + 1,  # + 1 car on passe la longueur comme scale aussi
        )

        outputs = self.get_output_modules(tasks, latent_dim, dim, RW_types)

        self.features_maker = TrajsFeatures()

        self.out_networks = {}
        self.losses = {}
        self.targets = {}
        for out in outputs:
            net, target, loss = outputs[out]
            self.out_networks[out] = net
            self.targets[out] = target
            self.losses[out] = loss
        self.out_networks = nn.ModuleDict(self.out_networks)
        self.loss_scale = defaultdict(lambda: 1.0 / 12.0)
        self.set_loss_scale()

        if "alpha" in tasks:
            self.MAE = MAE()
        if "model" in tasks:
            self.F1 = F1(len(self.hparams["RW_types"]))
        if "drift_norm" in tasks:
            self.EV = EV()

    def set_loss_scale(self):
        if "model" in self.losses:
            self.loss_scale["model"] = 1.0
        if "alpha" in self.losses:
            self.loss_scale["alpha"] = ((2.0 - 0.0) ** 2) / 12.0
        if "drift_norm" in self.losses:
            self.loss_scale["drift_norm"] = (0.5 ** 2) / 12
        if "drift" in self.losses:
            self.loss_scale["drift"] = 1.0
        if "force_norm" in self.losses:
            self.loss_scale["force_norm"] = (0.5 ** 2) / 12
        if "force" in self.losses:
            self.loss_scale["force"] = 2 * (80 ** 2) / 12
        if "log_tau" in self.losses:
            self.loss_scale["log_tau"] = 1.0 / 12.0
        if "log_diffusion" in self.losses:
            self.loss_scale["log_diffusion"] = ((-1 - (-3)) ** 2) / 12.0

    def get_output_modules(self, tasks, latent_dim, dim, RW_types):
        outputs = {}
        if "alpha" in tasks:
            outputs["alpha"] = (
                AlphaPredictor(input_dim=latent_dim),
                partial(self.get_target, target="alpha"),
                L2_loss,
            )
        if "model" in tasks:
            outputs["model"] = (
                MLP([latent_dim, 2 * latent_dim, latent_dim, len(RW_types)]),
                partial(self.get_target, target="model"),
                Category_loss,
            )
        if "drift" in tasks:
            outputs["drift"] = (
                MLP([latent_dim, 2 * latent_dim, latent_dim, dim]),
                partial(self.get_target, target="drift"),
                L2_loss,
            )
        if "drift_norm" in tasks:
            outputs["drift_norm"] = (
                MLP([latent_dim, 2 * latent_dim, 1], last_activation=nn.ReLU()),
                partial(self.get_target, target="drift_norm"),
                L2_loss,
            )
        if "force" in tasks:
            outputs["force"] = (
                MLP([latent_dim, 2 * latent_dim, latent_dim, dim]),
                partial(self.get_target, target="force"),
                L2_loss,
            )
        if "force_norm" in tasks:
            outputs["force_norm"] = (
                MLP([latent_dim, 2 * latent_dim, 1], last_activation=nn.ReLU()),
                partial(self.get_target, target="force_norm"),
                L2_loss,
            )
        if "log_theta" in tasks:
            outputs["log_theta"] = (
                MLP([latent_dim, 2 * latent_dim, latent_dim, 1]),
                partial(self.get_target, target="log_theta"),
                L2_loss,
            )
        if "log_tau" in tasks:
            outputs["log_tau"] = (
                MLP(
                    [latent_dim, 2 * latent_dim, latent_dim, latent_dim, 1],
                    out_range=(1, 2),
                ),
                partial(self.get_target, target="log_tau"),
                L2_loss,
            )
        if "log_diffusion" in tasks:
            outputs["log_diffusion"] = (
                MLP(
                    [latent_dim, 2 * latent_dim, latent_dim, latent_dim, 1],
                    out_range=(-4, 4),
                ),
                partial(self.get_target, target="log_diffusion"),
                L2_loss,
            )

        if len(outputs) == 0:
            print(tasks)
            raise BaseException("No outputs !")
        return outputs

    def get_target(self, data, target):
        if target == "alpha":
            return data.alpha
        elif target == "model":
            return data.model
        elif target == "drift":
            return data.drift
        elif target == "drift_norm":
            return data.drift_norm
        elif target == "force":
            return data.force
        elif target == "force_norm":
            return data.force_norm
        elif target == "log_theta":
            return data.log_theta
        elif target == "log_tau":
            return data.log_tau
        elif target == "log_diffusion":
            return data.log_diffusion
        else:
            raise NotImplementedError("Unknown target %s" % target)

    def forward(self, x):

        X, E, scales, orientation = self.features_maker(
            x, scale_types=self.hparams["scale_types"]
        )
        x.adj_t = x.adj_t.set_value(E)
        x.x = X
        x.scales = torch.cat([scales[k].view(-1, 1) for k in scales], dim=1)
        x.orientation = orientation
        # print(x.seed)
        # unique, counts = torch.unique(x.seed, return_counts=True)
        # print(unique)
        # print(counts)
        # Checked that counts is filled with ones, and that no ID appear in two different batches

        assert x.x.shape[1] == self.hparams["x_dim"]
        h = self.encoder(x)
        out = {}
        # targets = {}
        # losses = {}

        for net in self.out_networks:
            out[net] = self.out_networks[net](h)

        return out, h

    def training_step(self, batch, batch_idx):
        loss, out, targets = self.shared_step(batch, stage="train")
        return {"loss": loss, "preds": out, "targets": targets}

    def test_step(self, batch, batch_idx):
        loss, _, _ = self.shared_step(batch, stage="test")
        return loss

    def on_test_epoch_end(self):
        self.logger.log_hyperparams(self.hparams)

    def validation_step(self, batch, batch_idx):
        loss, out, targets = self.shared_step(batch, stage="val")
        trajs_info = targets.copy()
        del targets["length"]
        return {
            "loss": loss,
            "preds": out,
            "targets": targets,
            "trajs_info": trajs_info,
        }

    def shared_step(self, batch, stage="train"):
        # Part of the computation that is common to train, val and test
        out, h = self(batch)
        targets = {}
        losses = {}
        weights = {}

        # Save h ?
        # Callback ?
        # Write to TB

        targets["length"] = batch.length

        for net in out:

            targets[net] = self.targets[net](batch)
            w = is_concerned(targets[net])
            weights[net] = torch.mean(1.0 * w)
            # print(f"{net} : <w> = {torch.mean(1.*w)}")
            losses[net] = self.losses[net](out[net], targets[net], w)
            losses[net] /= self.loss_scale[net]
            self.log(
                "%s_%s_loss" % (net, stage),
                losses[net],
                on_step=False,
                on_epoch=True,
                logger=True,
            )

            if net == "alpha":
                self.log(
                    "%s_MAE" % stage,
                    self.MAE(out[net][w], targets[net][w]),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=stage == "val",
                    logger=True,
                )
            elif net == "model":
                self.log(
                    "%s_F1" % stage,
                    self.F1(out[net][w], targets[net][w].view(-1)),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=stage == "val",
                    logger=True,
                )
            elif net == "drift_norm":
                self.log(
                    "%s_EV" % stage,
                    self.EV(out[net][w], targets[net][w]),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=stage == "val",
                    logger=True,
                )

        out["latent"] = h
        # Pondération des loss en fonction du nombre de samples concernés
        loss = sum([losses[net] * weights[net] for net in losses])
        self.log(
            "%s_loss" % stage,
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        for key, value in out.items():
            out[key] = value.detach()

        return loss, out, targets

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), amsgrad=True, lr=self.hparams["lr"]
        )
        scheduler = ExponentialLR(optimizer, gamma=self.hparams["gamma"])
        return [optimizer], [scheduler]
