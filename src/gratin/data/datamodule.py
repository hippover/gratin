# import torch_geometric.transforms as Transforms
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
from torch.utils.data import random_split
from torch.utils.data.distributed import DistributedSampler
from . import default_ds_params, default_graph_info

# from ..simulation.diffusion_models import generators, params_sampler
from ..simulation.traj_tools import *

from .data_tools import *
from .dataset import *

EMPTY_FIELD_VALUE = -999


class DataModule(pl.LightningDataModule):
    def __init__(self, dl_params={}, ds_params={}, graph_info={}, no_parallel=False):

        super().__init__()

        self.ds_params = default_ds_params
        self.graph_info = default_graph_info
        self.ds_params.update(ds_params)
        self.graph_info.update(graph_info)
        self.dl_params = dl_params

        self.batch_size = dl_params["batch_size"]
        self.epoch_count = 0

        self.round = 0
        self.no_parallel = no_parallel
        self.exclude_keys = ["raw_positions"]

    def setup(self, stage=None, plot=True):
        if stage is None:
            print("stage is None, strange...")
        ds_params = {
            "N": self.ds_params["N"],
            "dim": self.ds_params["dim"],
            "noise_range": self.ds_params["noise_range"],
            "model_types": self.ds_params["RW_types"],
            "force_range": self.ds_params["force_range"],
            "logdiffusion_range": self.ds_params["logdiffusion_range"],
            "length_range": self.ds_params["length_range"],
            "time_delta": self.ds_params["time_delta"],
        }  # a bit redundant, but we recreate a ds_params, just to make sure it has only good arguments

        if stage == "fit" or stage is None:
            ds = TrajDataSet(
                **ds_params,
                graph_info=self.graph_info,
                seed_offset=self.ds_params["N"] * self.epoch_count,
            )
            self.ds_train, self.ds_val = random_split(
                ds, [19 * (self.ds_params["N"] // 20), 1 * (self.ds_params["N"] // 20)]
            )
        if stage == "test" or stage is None:
            ds = TrajDataSet(
                **ds_params,
                graph_info=self.graph_info,
                seed_offset=self.ds_params["N"] * (self.epoch_count + 1),
            )
            self.ds_test = ds

        data = ds[0]
        self.x_dim = data.x.shape[1]
        try:
            # self.e_dim = data.edge_attr.shape[1]
            self.e_dim = data.adj_t.dim()
        except:
            self.e_dim = 0
        if self.round == 0 and plot:
            print("Plotting")
            self.traj_examples = ds.make_plot()
        self.round += 1

    def train_dataloader(self):
        # print(f"Call train_dataloader for the {self.epoch_count}th time")
        self.setup(stage="fit")
        self.epoch_count += 1

        return DataLoader(
            self.ds_train,
            num_workers=self.dl_params["num_workers"],
            drop_last=True,
            batch_size=self.batch_size,
            exclude_keys=self.exclude_keys,
            # sampler=DistributedSampler(self.ds_train) if not self.no_parallel else None,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_val,
            num_workers=self.dl_params["num_workers"],
            drop_last=True,
            batch_size=self.batch_size,
            exclude_keys=self.exclude_keys,
            sampler=DistributedSampler(self.ds_val) if not self.no_parallel else None,
            pin_memory=True,
        )

    def test_dataloader(self, no_parallel=False):
        self.setup(stage="test")
        if no_parallel:
            # Otherwise, error Default process group is not initialized
            return DataLoader(
                self.ds_test,
                num_workers=self.dl_params["num_workers"],
                batch_size=self.batch_size,
                drop_last=True,
                exclude_keys=self.exclude_keys,
                pin_memory=True,
            )
        else:
            return DataLoader(
                self.ds_test,
                num_workers=self.dl_params["num_workers"],
                batch_size=self.batch_size,
                drop_last=True,
                exclude_keys=self.exclude_keys,
                sampler=DistributedSampler(self.ds_test),
                pin_memory=True,
            )