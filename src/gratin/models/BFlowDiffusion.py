from collections import defaultdict
from numpy.core.defchararray import array
import pytorch_lightning as pl
from pytorch_lightning.metrics.regression import MeanAbsoluteError as MAE
from pytorch_lightning.metrics.regression import MeanSquaredError as MSE
from pytorch_lightning.metrics import F1
from pytorch_lightning.metrics import ExplainedVariance as EV
import torch.nn as nn
import torch
from functools import partial
from ..layers.diverse import (
    batch_from_positions,
    batch_from_sub_batches,
    generate_batch_like,
)
from ..layers.features_init import *
from ..training.network_tools import L2_loss, Category_loss, is_concerned
from ..data.data_classes import DataModule
from ..layers.encoders import *
from ..layers.InvNet import InvertibleNet
from ..layers.fBMGenerator import fBMGenerator
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR
from itertools import chain
import os

DIFFUSION_TAG = "log_diffusion"
TAU_TAG = "log_tau"
ALPHA_TAG = "alpha"
VALID_TAGS = [DIFFUSION_TAG, TAU_TAG, ALPHA_TAG]


def get_T_values(
    T,
    num_lengths,
    vary_T=False,
    eval_mode=False,
    for_encoder=False,
    mix_lengths=False,
    log_sampling=True,
):
    if not vary_T:
        return [T]
    else:
        if not eval_mode:
            if not mix_lengths:
                # une seule longueur par batch
                if log_sampling:
                    return [
                        int(
                            np.power(
                                10,
                                np.random.uniform(
                                    low=np.log10(10 / 1.1), high=np.log10(1.1 * T)
                                ),
                            )
                        )
                    ]
                else:
                    return [int(np.random.uniform(low=10 / 1.1, high=1.1 * T))]
            else:
                # Les batches mélangent des longueurs, à la fin on vérifie si c'est OK
                OK = False
                while not OK:
                    if log_sampling:
                        T_values = [
                            np.power(
                                10,
                                np.random.uniform(
                                    low=np.log10(10.0 / 1.1), high=np.log10(1.1 * T)
                                ),
                            )
                            for _ in range(num_lengths)
                        ]
                        OK = np.abs(np.mean(np.log10(np.array(T_values))) - 2) < 0.1
                        OK = OK and (
                            np.quantile(np.array(T_values), 0.8)
                            > np.power(10, 1 + 0.7 * 2)
                        )
                        OK = OK and (
                            np.quantile(np.array(T_values), 0.2)
                            < np.power(10, 1 + 0.3 * 2)
                        )
                    else:
                        T_values = [
                            t
                            for t in np.random.uniform(
                                low=10.0 / 1.1, high=1.1 * T, size=num_lengths
                            )
                        ]
                        OK = np.abs(np.mean(np.array(T_values)) - 495) < T / 10
                        OK = OK and (np.quantile(np.array(T_values), 0.8) > 0.7 * T)
                        OK = OK and (np.quantile(np.array(T_values), 0.2) < 0.3 * T)
                T_values = [int(v) for v in T_values]
                return T_values
        else:
            return np.linspace(10, T, dtype=int, num=num_lengths, endpoint=True)


def reduce_alpha_range_if_needed(alpha_range, tau_range, generator, T_min, T_max):

    # On vérifie qu'aux extrémités des intervalles, la matrice de corrélation est positive
    OK = False
    T_values = [T_min, T_max]
    while not OK:
        a_min = torch.ones((1), device="cuda") * alpha_range[0]
        a_max = torch.ones((1), device="cuda") * alpha_range[1]

        tau_min = torch.ones((1), device="cuda") * tau_range[0]
        tau_max = torch.ones((1), device="cuda") * tau_range[1]
        diffusion = torch.ones_like(tau_min)

        try:
            for T in T_values:
                generator(a_min, tau_min, diffusion, T=T)
                print("T = %d |alpha = %.2f | tau = %.2f -> OK" % (T, a_min, tau_min))
                generator(a_min, tau_max, diffusion, T=T)
                print("T = %d |alpha = %.2f | tau = %.2f -> OK" % (T, a_min, tau_max))
        except Exception as e:
            alpha_range = (alpha_range[0] + 0.05, alpha_range[1])
            print(e)
            continue
        try:
            for T in T_values:
                generator(a_max, tau_min, diffusion, T=T)
                print("T = %d |alpha = %.2f | tau = %.2f -> OK" % (T, a_max, tau_min))
                generator(a_max, tau_max, diffusion, T=T)
                print("T = %d |alpha = %.2f | tau = %.2f -> OK" % (T, a_max, tau_max))
        except Exception as e:
            alpha_range = (alpha_range[0], alpha_range[1] - 0.05)
            print(e)
            continue
        OK = True
    a_range_end = alpha_range

    return a_range_end


class BFlowEncoder(pl.LightningModule):
    def __init__(
        self,
        n_c: int,  # number of convolutions
        latent_dim: int,
        dim: int,  # traj dim
        scale_types: list = ["step_var"],
        degree: int = 10,
        to_predict: str = ALPHA_TAG,
        alpha_range: tuple = (0.4, 1.6),
        tau_range: tuple = (5, 50),
        lr: float = 1e-3,
        n_lengths: int = 8,
        T: int = 200,
        vary_T: bool = True,
        vary_tau: bool = True,
    ):
        super().__init__()

        assert to_predict in VALID_TAGS, "%s is not a valid tag, use one of %s" % (
            to_predict,
            ", ".join(VALID_TAGS),
        )

        self.save_hyperparameters()
        x_dim = TrajsFeatures.x_dim(scale_types)
        e_dim = TrajsFeatures.e_dim(scale_types)
        self.save_hyperparameters({"x_dim": x_dim, "e_dim": e_dim})

        self.tau_range = self.hparams["tau_range"]
        if not self.hparams["vary_tau"]:
            self.tau_range = (self.hparams["T"], self.hparams["T"] + 1)
        self.alpha_range = self.hparams["alpha_range"]
        self.save_hyperparameters({"alpha_range": self.alpha_range})
        self.save_hyperparameters({"tau_range": self.tau_range})
        self.seed = 0

        self.generator = fBMGenerator(dim=self.hparams["dim"])

        self.features_maker = TrajsFeatures()

        self.summary_net = TrajsEncoder2(
            n_c=n_c,
            latent_dim=latent_dim,
            x_dim=self.hparams["x_dim"],
            e_dim=self.hparams["e_dim"],
            traj_dim=0,  # On se fiche de l'orientation
            n_scales=len(scale_types) + 1,  # + 1 because we add time as a scale
        )

        print("alpha range before test")
        print(self.alpha_range)
        self.alpha_range = reduce_alpha_range_if_needed(
            self.alpha_range, self.tau_range, self.generator, 10, self.hparams["T"]
        )
        print("alpha range after test")
        print(self.alpha_range)

        self.out_range = None
        if self.hparams["to_predict"] == ALPHA_TAG:
            self.out_range = (self.alpha_range[0] - 0.05, self.alpha_range[1] + 0.05)
        elif self.hparams["to_predict"] == TAU_TAG:
            self.out_range = (
                np.log10(self.tau_range[0]) - 0.05,
                np.log10(self.tau_range[1]) + 0.05,
            )
        self.simple_mlp = MLP([latent_dim, 128, 128, 1], out_range=self.out_range)

    def get_target(self, trajs):
        if self.hparams["to_predict"] == ALPHA_TAG:
            return trajs.alpha
        elif self.hparams["to_predict"] == DIFFUSION_TAG:
            return trajs.log_diffusion
        elif self.hparams["to_predict"] == TAU_TAG:
            return trajs.log_tau

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=self.hparams["lr"],
            amsgrad=True,
        )
        gamma = 0.99
        gen_lambda = (
            lambda step: max((gamma ** int(step // 250)), 0.01)
            # if step >= 1500
            # else 0.0
        )

        scheduler = {
            "scheduler": LambdaLR(optimizer, gen_lambda),
            "interval": "step",
        }
        return [optimizer], [scheduler]

    def get_new_batch_like(self, x, eval_mode):
        torch.manual_seed(self.seed)
        self.seed += x.alpha.shape[0]
        x = generate_batch_like(
            x,
            get_T_values(
                self.hparams["T"],
                self.hparams["n_lengths"],
                self.hparams["vary_T"],
                eval_mode=eval_mode,
                for_encoder=True,
            ),
            self.alpha_range,
            self.tau_range,
            self.generator,
            self.hparams["degree"],
            simulate_tau=self.hparams["vary_tau"],
            simulate_diffusion=True,  # Here, we should put False if we're in alpha-tau mode
        )
        assert torch.isnan(x.pos).sum() == 0
        return x

    def compute_features(self, x):
        X, E, scales, orientation = self.features_maker(
            x, scale_types=self.hparams["scale_types"]
        )
        x.adj_t = x.adj_t.set_value(E)
        x.x = X
        x.scales = torch.cat([scales[k].view(-1, 1) for k in scales], dim=1)
        # x.log_diffusion_correction = torch.clamp(
        #    x.log_diffusion - torch.log10(scales["step_var"]).view(-1, 1), -1, 1
        # )

        assert x.x.shape[1] == self.hparams["x_dim"]
        return x

    def forward(self, x, return_input=True, eval_mode=False, return_latent=False):
        # Si x ne contient pas de trajectoires, on en génère
        if torch.isnan(x.pos).sum() > 0:
            # print(self.seed)
            x = self.get_new_batch_like(x, eval_mode=eval_mode)

        # NORMAL FORWARD PASS
        x = self.compute_features(x)
        h = self.summary_net(x)
        output = self.simple_mlp(h)

        if return_latent:
            return h, x

        if not return_input:
            return output
        else:
            return output, x

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        output, trajs = self(batch, return_input=True)
        target = self.get_target(trajs)
        loss = torch.mean((output - target) ** 2)
        MAE_loss = torch.mean(torch.abs(output - target))
        self.log(
            "training_loss", value=loss, on_step=True, on_epoch=True, prog_bar=True
        )
        self.log(
            "training_MAE_%s" % self.hparams["to_predict"],
            value=MAE_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        assert torch.sum(torch.isnan(trajs.alpha)) == 0
        return {
            "loss": loss,
            "trajs_info": {
                "length": trajs.length,
                "alpha": trajs.alpha,
                "log_tau": trajs.log_tau,
                "log_diffusion": trajs.log_diffusion,
            },
            "targets": {self.hparams["to_predict"]: target},
            "preds": {self.hparams["to_predict"]: output},
        }

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        output, trajs = self(batch, return_input=True, eval_mode=True)
        target = self.get_target(trajs)
        loss = torch.mean((output - target) ** 2)
        MAE_loss = torch.mean(torch.abs(output - target))
        self.log("val_L2", value=loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val_MAE_%s" % self.hparams["to_predict"],
            value=MAE_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        assert torch.sum(torch.isnan(trajs.alpha)) == 0

        return {
            "loss": loss,
            "trajs_info": {
                "length": trajs.length,
                "alpha": trajs.alpha,
                "log_tau": trajs.log_tau,
                "log_diffusion": trajs.log_diffusion,
            },
            "targets": {self.hparams["to_predict"]: target},
            "preds": {self.hparams["to_predict"]: output},
        }


class BFlowMain(pl.LightningModule):
    def __init__(
        self,
        encoders_paths: dict,
        to_infer: list,
        gamma: float = 0.98,
        lr: float = 1e-3,
        n_lengths: int = 32,
        n_ACBs: int = 3,
        alpha_clip: float = 1.7,
        couple_factor: float = 1.5,
        t_abs: float = 4,
        s_abs: float = 4,
        log_sampling: bool = True,
        mix_lengths: bool = True,
        **kwargs
    ):

        super().__init__()
        self.save_hyperparameters()
        self.save_hyperparameters(self.load_summary_nets(encoders_paths, to_infer))

        self.metrics = {
            ALPHA_TAG: MAE(),
            TAU_TAG: MSE(),
            DIFFUSION_TAG: MSE(),
        }

        self.generator = fBMGenerator(dim=self.hparams["dim"])
        self.features_maker = TrajsFeatures()
        self.global_step_last_T_generated = -1

        # Removed, slows training too much
        # self.trainable_summary_net = TrajsEncoder2(
        #    traj_dim=0,  # Orientation non prise en compte,
        #    n_c=16,
        #    x_dim=self.hparams["x_dim"],
        #    e_dim=self.hparams["e_dim"],
        #    n_scales=len(self.hparams["scale_types"]) + 1,
        #    latent_dim=self.hparams["latent_dim"],
        # )

        self.invertible_net = InvertibleNet(
            dim_theta=self.dim_theta,
            dim_x=self.hparams["latent_dim"]
            * len(self.hparams["to_infer"]),  # latent_dim dimensions per encoder
            n_blocks=n_ACBs,
            alpha_clip=alpha_clip,
            t_abs=self.hparams["t_abs"],
            s_abs=self.hparams["s_abs"],
        )

        self.norm_dist = torch.distributions.normal.Normal(0.0, 1.0, validate_args=None)
        self.unif_dist = torch.distributions.uniform.Uniform(-0.5, 0.5)

        print(self.hparams)

    def load_summary_nets(self, encoders_paths, to_infer):
        self.summary_nets = {}

        encoders = {}
        for param in to_infer:
            assert param in encoders_paths, "no path for %s" % param
            assert os.path.exists(
                encoders_paths[param]
            ), "file does not exist (%s) : %s" % (param, encoders_paths[param])
            encoders[param] = BFlowEncoder.load_from_checkpoint(encoders_paths[param])

            # encoders[param] = encoders[param].to(self.device)
            encoders[param].eval()
            encoders[param].freeze()

        ## Update hyperparameters and check that encoders have the same settings
        ## Store encoders
        encoder_values = defaultdict(lambda: [])
        hparams_to_check = [
            "T",
            "latent_dim",
            "tau_range",
            "alpha_range",
            "degree",
            "dim",
            "scale_types",
            "x_dim",
            "e_dim",
            "vary_tau",
            "vary_T",
        ]
        for param in to_infer:
            assert param in encoders, "%s has no encoder" % param
            for hparam in hparams_to_check:
                value = encoders[param].hparams[hparam]
                if type(value) is list:
                    value = ",".join(value)
                encoder_values[hparam].append(value)
            self.summary_nets[param] = encoders[param].summary_net

        self.summary_nets = nn.ModuleDict(self.summary_nets)
        hparams_to_save = {}

        for hparam in hparams_to_check:
            assert (
                len(set(encoder_values[hparam])) == 1
            ), "conflict between task-specific summary nets -> more than 1 %s value : [%s]" % (
                hparam,
                ", ".join(encoder_values[hparam]),
            )
            value = encoder_values[hparam][0]
            if type(value) is str:
                if "," in value:
                    value = value.split(",")

            try:
                hparams_to_save.update({hparam: value})
            except Exception as e:
                print("Could not save param %s with value %s" % (hparam, value))
                raise

        self.dim_theta = len(to_infer)
        assert self.dim_theta >= 2

        return hparams_to_save

    def scale(self, param, range, inverse):
        # bypass, no scaling
        # return param
        m, M = range
        # To avoid infinity, we slightly widen the range
        range_size = M - m
        m -= 0.05 * range_size
        M += 0.05 * range_size
        if inverse == False:
            # return self.unif_dist.icdf((param - m) / (M - m))
            return self.norm_dist.icdf((param - m) / (M - m))
        elif inverse == True:
            # return self.unif_dist.cdf(param) * (M - m) + m
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

    def couple(self, z, inverse=False):
        # Made sure that it is well inverted

        r = torch.sqrt(torch.sum(z ** 2, dim=1))
        angle = r * self.hparams["couple_factor"]
        if inverse:
            angle *= -1
        rot_matrix_1 = torch.stack([torch.cos(angle), torch.sin(angle)], dim=1)
        rot_matrix_2 = torch.stack([-torch.sin(angle), torch.cos(angle)], dim=1)
        rot_matrix = torch.stack([rot_matrix_1, rot_matrix_2], dim=0)

        res = rot_matrix @ z.T

        z_rotated = torch.stack([torch.diag(res[0]), torch.diag(res[1])], dim=1)

        return z_rotated

    def make_theta(self, x):

        theta_dims = []
        if ALPHA_TAG in self.hparams["to_infer"]:
            theta_dims.append(self.scale_alpha(x.alpha).float().view(-1, 1))
        if DIFFUSION_TAG in self.hparams["to_infer"]:
            theta_dims.append(self.scale_logdiff(x.log_diffusion).float().view(-1, 1))
        if TAU_TAG in self.hparams["to_infer"]:
            theta_dims.append(self.scale_logtau(x.log_tau).float().view(-1, 1))
        theta = torch.cat(theta_dims, dim=1)

        # Couple variables
        theta = self.couple(theta)
        return theta

    def get_params(self, theta):

        # decouple variables
        theta = self.couple(theta, inverse=True)
        params = {}
        if ALPHA_TAG in self.hparams["to_infer"]:
            params[ALPHA_TAG] = self.scale_alpha(
                theta[:, len(params)].view(-1, 1), inverse=True
            )
        if DIFFUSION_TAG in self.hparams["to_infer"]:
            params[DIFFUSION_TAG] = self.scale_logdiff(
                theta[:, len(params)].view(-1, 1), inverse=True
            )
        if TAU_TAG in self.hparams["to_infer"]:
            params[TAU_TAG] = self.scale_logtau(
                theta[:, len(params)].view(-1, 1), inverse=True
            )
        return params

    def get_new_batch_like(self, x, eval_mode):

        # This is to get the same T_values across batches when we accumulate gradients
        T_values_args = {
            "T": self.hparams["T"],
            "num_lengths": self.hparams["n_lengths"],
            "vary_T": False if TAU_TAG in self.hparams["to_infer"] else True,
            "eval_mode": eval_mode,
            "mix_lengths": self.hparams["mix_lengths"],
            "log_sampling": self.hparams["log_sampling"],
        }
        if eval_mode or not self.hparams["mix_lengths"]:
            T_values = get_T_values(**T_values_args)
        else:
            if self.global_step > self.global_step_last_T_generated:
                T_values = get_T_values(**T_values_args)
                self.global_step_last_T_generated = self.global_step
                self.last_T_values = T_values
            else:
                T_values = self.last_T_values
        x = generate_batch_like(
            x,
            T_values,
            self.hparams["alpha_range"],
            self.hparams["tau_range"],
            self.generator,
            self.hparams["degree"],
            simulate_tau=TAU_TAG in self.hparams["to_infer"],
        )

        assert torch.isnan(x.pos).sum() == 0
        return x

    def compute_features(self, x):
        X, E, scales, orientation = self.features_maker(
            x, scale_types=self.hparams["scale_types"]
        )
        x.adj_t = x.adj_t.set_value(E)
        x.x = X
        x.scales = torch.cat([scales[k].view(-1, 1) for k in scales], dim=1)
        # x.log_diffusion_correction = torch.clamp(
        #    x.log_diffusion - torch.log10(scales["step_var"]).view(-1, 1), -1, 1
        # )

        assert x.x.shape[1] == self.hparams["x_dim"]
        return x

    def get_latent_vector(self, x):
        with torch.no_grad():
            h = torch.cat(
                [self.summary_nets[param](x) for param in self.hparams["to_infer"]],
                dim=1,
            )
        # h = torch.cat([h, self.trainable_summary_net(x)], dim=1)
        return h

    def forward(
        self,
        x,
        sample=False,
        n_repeats=1,
        batch_idx=0,
        return_input=False,
        eval_mode=False,
    ):

        # Si x ne contient pas de trajectoires, on en génère
        if torch.isnan(x.pos).sum() > 0:
            x = self.get_new_batch_like(x, eval_mode=eval_mode)

        x = self.compute_features(x)
        true_theta = self.make_theta(x)
        h = self.get_latent_vector(x)
        # Checked that with trained sumary nets h has a variance of 1

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

        # self.invertible_net.set_s_lim(self.global_step / 3000.0)
        self.invertible_net.set_s_lim(4)
        z, log_J = self(batch, batch_idx=batch_idx)
        z_norm2 = torch.sum(z ** 2, dim=1)
        l = torch.mean(0.5 * z_norm2 - log_J, dim=0)
        self.log("training_loss", value=l, on_step=False, on_epoch=True, prog_bar=True)
        self.log("z_norm", value=torch.mean(z_norm2), on_step=True, prog_bar=True)
        self.log("log_J", value=torch.mean(log_J), on_step=True, prog_bar=True)

        return {"loss": l}

    def sample_step(self, batch, batch_idx, n_repeats=20):
        theta, true_theta, trajs = self(
            batch,
            batch_idx=batch_idx,
            sample=True,
            n_repeats=n_repeats,
            eval_mode=True,
            return_input=True,
        )

        true_theta = true_theta.repeat_interleave(n_repeats, 0)
        preds = self.get_params(theta)
        targets = self.get_params(true_theta)

        for param in self.hparams["to_infer"]:
            self.metrics[param](preds[param], targets[param])

        return preds, targets, trajs

    def on_pretrain_routine_start(self):
        print("Pretrain routine")
        print("Device is %s" % self.device)
        for param in self.metrics:
            self.metrics[param] = self.metrics[param].to(self.device)

    def log_metrics(self, step="test"):
        for param in self.hparams["to_infer"]:
            self.log(
                "hp/%s_%s_metric" % (param, step),
                self.metrics[param],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

    def test_step(self, batch, batch_idx):
        preds, targets, trajs = self.sample_step(batch, batch_idx=batch_idx)
        self.log_metrics(step="test")
        return {"targets": targets, "preds": preds}

    def on_train_start(self):
        metrics_defaults = {}
        for param in self.hparams["to_infer"]:
            metrics_defaults["hp/%s_test_metric" % param] = np.inf
            metrics_defaults["hp/%s_val_metric" % param] = np.inf
        self.logger.log_hyperparams(self.hparams, metrics_defaults)

    def on_validation_epoch_end(self):
        print("Validation epoch end")
        self.logger.experiment.flush()

    def validation_step(self, batch, batch_idx):
        preds, targets, trajs = self.sample_step(batch, batch_idx=batch_idx)
        self.log_metrics(step="val")
        return {
            "targets": targets,
            "trajs_info": {
                "length": trajs.length,
                "alpha": trajs.alpha,
                "log_tau": trajs.log_tau,
                "log_diffusion": trajs.log_diffusion,
            },
            "preds": preds,
        }

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(
            self.invertible_net.parameters(),
            lr=self.hparams["lr"],
            amsgrad=True,
        )

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                patience=4,
                min_lr=1e-4,
                cooldown=4,
                verbose=True,
                factor=0.2,
            ),
            "interval": "epoch",
            "monitor": "training_loss",
        }

        return [optimizer], [scheduler]
