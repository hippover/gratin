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
    def __init__(self, dl_params={}, ds_params={}, graph_info={}):

        super().__init__()

        self.ds_params = default_ds_params
        self.graph_info = default_graph_info
        self.ds_params.update(ds_params)
        self.graph_info.update(graph_info)
        self.dl_params = dl_params

        self.batch_size = dl_params["batch_size"]
        self.epoch_count = 1

        self.round = 0
        self.exclude_keys = ["raw_positions"]

    def setup(self, stage=None, plot=True):
        if stage is None:
            print("stage is None, strange...")
        self.recreate_datasets(stage=stage, plot=plot)

    def recreate_datasets(self, stage="fit", plot=True):
        ds_params = {
            "N": self.ds_params["N"],
            "dim": self.ds_params["dim"],
            "noise_range": self.ds_params["noise_range"],
            "model_types": self.ds_params["RW_types"],
            "logdiffusion_range": self.ds_params["logdiffusion_range"],
            "length_range": self.ds_params["length_range"],
            "time_delta_range": self.ds_params["time_delta_range"],
        }  # a bit redundant, but we recreate a ds_params, just to make sure it has only good arguments

        if stage == "fit" or stage is None:
            self.ds_train = TrajDataSet(
                **ds_params,
                graph_info=self.graph_info,
                # seed_offset=0
                seed_offset=self.ds_params["N"] * self.epoch_count,
            )
        if self.round == 0:
            ds_params_val = dict(ds_params)
            ds_params_val["N"] = 3000
            self.ds_val = TrajDataSet(
                **ds_params_val,
                graph_info=self.graph_info,
                seed_offset=0,
            )

        if stage == "test" or stage is None:
            ds_params_test = dict(ds_params)
            ds_params_test["N"] //= 10
            ds = TrajDataSet(
                **ds_params_test,
                graph_info=self.graph_info,
                # seed_offset=0
                seed_offset=self.ds_params["N"] * (self.epoch_count + 1),
            )
            self.ds_test = ds

        if self.round == 0 and plot:
            print("Plotting")
            self.traj_examples = ds.make_plot()
        self.round += 1

    def train_dataloader(self):
        # print(f"Call train_dataloader for the {self.epoch_count}th time")
        self.epoch_count += 1
        self.recreate_datasets(stage="fit")

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
            # sampler=DistributedSampler(self.ds_val) if not self.no_parallel else None,
            pin_memory=True,
        )

    def test_dataloader(self):
        self.recreate_datasets(stage="test")

        return DataLoader(
            self.ds_test,
            num_workers=self.dl_params["num_workers"],
            batch_size=self.batch_size,
            drop_last=True,
            exclude_keys=self.exclude_keys,
            # sampler=DistributedSampler(self.ds_test),
            pin_memory=True,
        )
