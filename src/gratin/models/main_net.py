from collections import defaultdict
import pytorch_lightning as pl
from torchmetrics import MeanAbsoluteError as MAE
from torchmetrics.classification import MulticlassF1Score as F1
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


class MainNet(pl.LightningModule):
    def __init__(
        self,
        n_c: int,  # number of convolutions
        latent_dim: int,
        dim: int,  # traj dim
        gamma: float = 0.98,
        lr: float = 1e-3,
        log_distance_features: bool = True,
        RW_types: list = ["fBM"],
        scale_types: list = ["step_std"],
        predict_alpha: bool = True,
        predict_model: bool = True
    ):
        super().__init__()
        
        assert predict_alpha or predict_model

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

        outputs = {}
        if predict_alpha:
            outputs["alpha"] = (
                AlphaPredictor(input_dim=latent_dim),
                partial(self.get_target, target="alpha"),
                # L2_loss,
                nn.MSELoss(),
            )
        if predict_model:
            outputs["model"] = (
                MLP([latent_dim, 2 * latent_dim, latent_dim, len(RW_types)]),
                partial(self.get_target, target="model"),
                # Category_loss,
                nn.CrossEntropyLoss(),
            )

        self.features_maker = TrajsFeaturesSimple()

        self.out_networks = {}
        self.losses = {}
        self.targets = {}
        for out in outputs:
            net, target, loss = outputs[out]
            self.out_networks[out] = net
            self.targets[out] = target
            self.losses[out] = loss
        self.losses = nn.ModuleDict(self.losses)
        self.out_networks = nn.ModuleDict(self.out_networks)
        self.loss_scale = defaultdict(lambda: 1.0 / 12.0)
        self.set_loss_scale()

        self.MAE_train = MAE()
        self.MAE_val = MAE()
        self.MAE_test = MAE()
        self.F1_train = F1(len(self.hparams["RW_types"]))
        self.F1_val = F1(len(self.hparams["RW_types"]))
        self.F1_test = F1(len(self.hparams["RW_types"]))
        self.metrics = {}
        self.metrics[("alpha", "train")] = self.MAE_train
        self.metrics[("model", "train")] = self.F1_train
        self.metrics[("alpha", "val")] = self.MAE_val
        self.metrics[("model", "val")] = self.F1_val
        self.metrics[("alpha", "test")] = self.MAE_test
        self.metrics[("model", "test")] = self.F1_test

    def set_loss_scale(self):
        if "model" in self.losses:
            self.loss_scale["model"] = np.log(len(self.hparams["RW_types"]))
        if "alpha" in self.losses:
            self.loss_scale["alpha"] = scipy.stats.uniform.var(loc=0, scale=2)

    def get_target(self, data, target):
        if target == "alpha":
            return data.alpha
        elif target == "model":
            return data.model.view(
                -1,
            )
        else:
            raise NotImplementedError("Unknown target %s" % target)

    def forward(self, x):

        X, E, scales = self.features_maker(
            x,
            # scale_types=self.hparams["scale_types"],
            # log_distance_features=self.hparams["log_distance_features"],
        )
        x.adj_t = x.adj_t.set_value(E, layout="coo")
        x.x = X
        x.scales = torch.cat(
            [scales[k].view(-1, 1) for k in self.hparams["scale_types"]], dim=1
        )

        assert x.x.shape[1] == self.hparams["x_dim"]
        h = self.encoder(x)
        out = {}

        for net in self.out_networks:
            out[net] = self.out_networks[net](h)

        return out, h

    # def on_fit_start(self):
    #    print("Self.device")
    #    print(self.device)
    #    for key, v in self.metrics.items():
    #        v = v.to(self.device)

    def training_step(self, batch, batch_idx):
        loss, out, targets = self.shared_step(batch, stage="train")
        return loss
        # return {"loss": loss, "preds": out, "targets": targets}

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

        targets["length"] = batch.length

        for net in out:

            targets[net] = self.targets[net](batch)
            w = is_concerned(targets[net])
            weights[net] = torch.mean(1.0 * w)
            # print(f"{net} : <w> = {torch.mean(1.*w)}")
            # losses[net] = self.losses[net](out[net], targets[net], w)
            losses[net] = self.losses[net](out[net][w], targets[net][w])
            losses[net] /= self.loss_scale[net]
            self.log(
                "%s_%s_loss" % (net, stage),
                losses[net],
                on_step=False,
                on_epoch=True,
                logger=True,
                batch_size=h.shape[0],
            )

            self.metrics[(net, stage)](out[net][w], targets[net][w])
            self.log(
                "%s_%s_%s"
                % (net, stage, self.metrics[(net, stage)].__class__.__qualname__),
                self.metrics[(net, stage)],
                on_step=False,
                on_epoch=True,
                prog_bar=stage != "val",
                logger=True,
                batch_size=h.shape[0],
            )

        out["latent"] = h
        # Pondération des loss en fonction du nombre de samples concernés
        loss = sum([losses[net] * weights[net] for net in losses])
        self.log(
            "%s_loss" % stage,
            loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=h.shape[0],
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
