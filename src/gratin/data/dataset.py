import torch_geometric.transforms as Transforms
from torch_geometric.data import Dataset
import numpy as np
from ..simulation.diffusion_models import generators, params_sampler
from ..simulation.traj_tools import *
from ..data.data import TrajData
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import truncnorm
import logging
from scipy.stats import truncexpon

EMPTY_FIELD_VALUE = -999


class TrajDataSet(Dataset):
    def __init__(
        self,
        N: int,
        dim: int,
        graph_info: dict,
        length_range: tuple,  # e.g = (5,50)
        noise_range: tuple,  # e.g. = (0.1,0.5),
        model_types: list,  # e.g. = ["fBM","CTRW"],
        seed_offset: int,  # e.g. = 0):
        time_delta_range: tuple,
        logdiffusion_range: tuple = (-2.0, 0.5),
        max_blinking_fraction: float = 0.2,
    ):
        self.N = N

        self.seed_offset = seed_offset
        logging.debug("Create TrajDataset, seed_offset = %d" % seed_offset)
        self.graph_info = graph_info

        self.generators = generators[dim]
        self.model_types = model_types
        self.length_range = length_range
        self.noise_range = noise_range
        self.dim = dim
        self.log_diff_range = logdiffusion_range  # (-2.0, 0.5)
        self.time_delta_range = time_delta_range
        self.max_blinking_fraction = max_blinking_fraction
        super(TrajDataSet, self).__init__(transform=Transforms.ToSparseTensor())

    def len(self):
        return self.N

    def generate_log_diffusion(self):
        # log_diffusion = np.random.uniform(*self.log_diff_range)
        if self.log_diff_range[0] == self.log_diff_range[1]:
            return self.log_diff_range[0]
        else:
            OK = False
            diff_mean = 0.5 * (self.log_diff_range[0] + self.log_diff_range[1])
            diff_std = 0.3 * (self.log_diff_range[1] - self.log_diff_range[0])
            a, b = (self.log_diff_range[0] - diff_mean) / diff_std, (
                self.log_diff_range[1] - diff_mean
            ) / diff_std
            while not OK:
                log_diffusion = truncnorm.rvs(
                    a,
                    b,
                    loc=diff_mean,
                    scale=diff_std,
                )
                OK = (log_diffusion >= self.log_diff_range[0]) and (
                    log_diffusion <= self.log_diff_range[1]
                )
            return log_diffusion

    def get_traj_params(self):
        model_index = np.random.choice(len(self.model_types))
        model = self.model_types[model_index]
        params = params_sampler(model, seed=None)
        # length = np.random.randint(low=self.length_range[0], high=self.length_range[1])
        """
        length = int(
            np.power(
                10,
                np.random.uniform(
                    low=np.log10(self.length_range[0]),
                    high=np.log10(self.length_range[1]),
                ),
            )
        )
        """
        max_length = self.length_range[1]
        scale = max_length / 8.0
        min_length = self.length_range[0]
        length = int(
            min_length
            + truncexpon.rvs(scale=scale, b=(max_length - min_length) / scale)
        )

        log_diffusion = self.generate_log_diffusion()

        diffusion = np.power(10.0, log_diffusion)

        pos_uncertainty = np.random.uniform(self.noise_range[0], self.noise_range[1])
        # noise_factor = np.random.uniform(self.noise_range[0], self.noise_range[1])

        time_step_min, time_step_max = self.time_delta_range
        time_step = np.power(
            10.0,
            np.random.uniform(
                low=np.log10(time_step_min), high=np.log10(time_step_max)
            ),
        )

        trajs_params = {
            "model_index": model_index,
            "model": model,
            "params": params,
            "length": length,
            "diffusion": diffusion,
            "log_diffusion": log_diffusion,
            "pos_uncertainty": pos_uncertainty,
            "time_step": time_step,
        }

        return trajs_params

    def get_raw_traj(self, traj_params):

        OK = False
        counter = 0

        while not OK:
            raw_pos = self.generators[traj_params["model"]](
                T=traj_params["length"], **traj_params["params"]
            )
            raw_pos -= raw_pos[0]
            OK = np.isnan(raw_pos).sum() == 0
            # OK = OK and (np.max(raw_pos[:, 0]) > np.min(raw_pos[:, 0]))
            OK = OK or (traj_params["model"] == "empty")
            if not OK:
                counter += 1
                # print("raw_pos has nans %d" % counter)
                # print(params)
        if counter > 10:
            # print(
            #    "More than 10 trials (%d) to generate a %s"
            #    % (counter, traj_params["model"])
            # )
            # for p in traj_params["params"]:
            #    print(p, traj_params["params"][p])
            pass

        assert raw_pos is not None, "raw_pos is None %s" % raw_pos

        return raw_pos

    def apply_diffusion_and_noise(self, raw_pos, traj_params):

        diffusion = traj_params["diffusion"]
        # force_norm = EMPTY_FIELD_VALUE

        raw_pos *= np.sqrt(diffusion * traj_params["time_step"])

        if traj_params["pos_uncertainty"] > 0.0:
            noisy_pos = raw_pos + traj_params["pos_uncertainty"] * np.random.randn(
                *raw_pos.shape
            )
        else:
            noisy_pos = raw_pos
        return noisy_pos

    def get_traj(self, seed, return_raw=False):

        # print("Get traj %d" % seed)

        np.random.seed(seed)
        traj_params = self.get_traj_params()
        traj_params["seed"] = seed
        # print(traj_params)
        # print(traj_params["log_diffusion"])
        raw_pos = self.get_raw_traj(traj_params)
        pos = self.apply_diffusion_and_noise(raw_pos, traj_params)

        assert pos is not None, "pos is None %s" % traj_params

        if not return_raw:
            return pos, traj_params
        else:
            return pos, traj_params, raw_pos

    def make_plot(self):
        fig = plt.figure(figsize=(10, 10), dpi=150)

        traj_params = {"length": 20, "diffusion": 1e-3, "pos_uncertainty": 0.1}

        for i in range(10):

            for j in range(10):
                if i <= 3:
                    traj_params["model"] = "BM"
                    traj_params["diffusion"] = np.power(10.0, self.log_diff_range[1])
                    traj_params["params"] = {"alpha": 1.0}
                    traj_params["pos_uncertainty"] = 0.0
                else:
                    traj_params = self.get_traj_params()

                model = traj_params["model"]
                raw_pos = self.get_raw_traj(traj_params)
                noisy_pos = self.apply_diffusion_and_noise(raw_pos, traj_params)

                ax = fig.add_subplot(10, 10, j * 10 + i + 1)
                if self.dim == 1:
                    ax.plot(noisy_pos[:, 0])
                else:
                    ax.plot(noisy_pos[:, 0], noisy_pos[:, 1])
                params = traj_params["params"]
                pos_uncertainty = traj_params["pos_uncertainty"]
                if "tau" in params:
                    desc = "$\\tau_c$ = %d" % params["tau"]
                elif "alpha" in params:
                    desc = f"{model[:4]} - $\\alpha$:{float(params['alpha']):{1}.{2}} - N{pos_uncertainty:{1}.{1}}"
                else:
                    desc = f"{model[:4]} - N{pos_uncertainty:{1}.{1}}"
                ax.set_title(desc, fontsize=6)
                if self.dim >= 2:
                    ax.set_aspect("equal")
                plt.axis("off")
        plt.tight_layout()
        return fig

    def get(self, idx):
        # pos = np.zeros((10,2))

        # if self.train:
        #    print("Train index %d" % idx)
        # else:
        #    print("Test index %d" % idx)
        # print("Get idx = %d" % idx)
        seed = idx + self.seed_offset
        # print("generating seed = %d " % seed)
        noisy_pos, traj_params, raw_pos = self.get_traj(seed=seed, return_raw=True)
        assert noisy_pos is not None, "noisy_pos is None : %s" % traj_params
        traj_info = traj_params["params"]
        traj_params.pop("params")
        traj_info.update(traj_params)
        # print("traj_info : %d" % traj_info["seed"])

        time = np.arange(noisy_pos.shape[0]) * traj_info["time_step"]

        L = noisy_pos.shape[0]
        if L > 7 and self.max_blinking_fraction > 0:
            n_points_to_remove = np.random.randint(
                0, int(self.max_blinking_fraction * L) + 1
            )
        else:
            n_points_to_remove = 0
        to_keep = np.random.choice(L, size=L - n_points_to_remove, replace=0)
        to_keep_bool = np.isin(np.arange(L), to_keep)

        noisy_pos = noisy_pos[to_keep_bool]
        raw_pos = raw_pos[to_keep_bool]
        time = time[to_keep_bool]

        return TrajData(
            raw_positions=noisy_pos,
            time=time,
            graph_info=self.graph_info,
            traj_info=traj_info,
            original_positions=raw_pos,
        )


class ExpTrajDataSet(Dataset):
    def __init__(
        self,
        dim: int,
        graph_info: dict,
        trajs: list,
        times: list,
    ):  # e.g. = 0):
        self.N = len(trajs)
        self.trajs = trajs
        self.times = times
        self.graph_info = graph_info
        self.dim = dim

        transform = Transforms.ToSparseTensor()

        super(ExpTrajDataSet, self).__init__(transform=transform)

    def len(self):
        return self.N

    def make_plot(self, info=None):
        fig = plt.figure(figsize=(10, 10), dpi=150)

        N = 100
        if info is not None:
            N = len(info)
        for i in range(min(N, len(self))):
            noisy_pos = self.trajs[i]
            ax = fig.add_subplot(10, 10, i + 1)
            if self.dim == 1:
                ax.plot(noisy_pos[:, 0])
            else:
                ax.plot(noisy_pos[:, 0], noisy_pos[:, 1])
            ax.set_aspect("equal")
            if info is not None:
                ax.set_title("info : %.2f" % info[i])
            plt.axis("off")
        plt.tight_layout()
        return fig

    def get(self, idx):

        noisy_pos = self.trajs[idx]
        time = self.times[idx]
        traj_info = {"index": idx, "pos_uncertainty": 0.03, "seed": idx}
        return TrajData(
            noisy_pos,
            time=time,
            graph_info=self.graph_info,
            traj_info=traj_info,
            original_positions=None,
        )
