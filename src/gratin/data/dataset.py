import torch_geometric.transforms as Transforms
from torch_geometric.data import Dataset
import numpy as np
import warnings
from ..simulation.diffusion_models import generators, params_sampler
from ..simulation.traj_tools import *
from ..data.data import TrajData
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import truncnorm

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
        time_delta: float,
        logdiffusion_range: tuple = (-2.0, 0.5),
    ):
        self.N = N

        self.seed_offset = seed_offset
        print("Create TrajDataset, seed_offset = %d" % seed_offset)
        self.graph_info = graph_info

        self.generators = generators[dim]
        self.model_types = model_types
        self.length_range = length_range
        self.noise_range = noise_range
        self.dim = dim
        self.log_diff_range = logdiffusion_range  # (-2.0, 0.5)
        self.time_delta = time_delta
        super(TrajDataSet, self).__init__(transform=Transforms.ToSparseTensor())

    def len(self):
        return self.N

    def generate_log_diffusion(self):
        OK = False
        # log_diffusion = np.random.uniform(*self.log_diff_range)
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
        length = int(
            np.power(
                10,
                np.random.uniform(
                    low=np.log10(self.length_range[0]),
                    high=np.log10(self.length_range[1]),
                ),
            )
        )

        log_diffusion = self.generate_log_diffusion()

        diffusion = np.power(10.0, log_diffusion)

        pos_uncertainty = np.random.uniform(self.noise_range[0], self.noise_range[1])
        # noise_factor = np.random.uniform(self.noise_range[0], self.noise_range[1])

        trajs_params = {
            "model_index": model_index,
            "model": model,
            "params": params,
            "length": length,
            "diffusion": diffusion,
            "log_diffusion": log_diffusion,
            "pos_uncertainty": pos_uncertainty,
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

        raw_pos *= np.sqrt(diffusion * self.time_delta)

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
        return TrajData(
            noisy_pos,
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
    ):  # e.g. = 0):
        self.N = len(trajs)
        self.trajs = trajs
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
        traj_info = {"index": idx, "pos_uncertainty": 0.03, "seed": idx}
        return TrajData(
            noisy_pos,
            graph_info=self.graph_info,
            traj_info=traj_info,
            original_positions=None,
        )
