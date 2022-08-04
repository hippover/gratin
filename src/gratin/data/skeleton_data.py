import numpy as np
from ..simulation.traj_tools import *
from torch_geometric.data import Data
import torch
import numpy as np

from .data_tools import *

EMPTY_FIELD_VALUE = np.nan


class SkeletonTrajData(Data):
    def __init__(
        self,
        raw_positions,
        time,
        graph_info={},
        traj_info={},
        original_positions=None,
        **kwargs,
    ):
        try:
            dim = raw_positions.shape[1]

            if original_positions is None:
                original_positions = raw_positions
            if not type(original_positions) is np.array:
                original_positions = np.array(original_positions).copy()

            default_traj_info = {
                "model": "unknown",
                "model_index": 0,
                "log_theta": EMPTY_FIELD_VALUE,
                "alpha": EMPTY_FIELD_VALUE,
                "noise": 0.0,
                "tau": np.inf,
                "seed": EMPTY_FIELD_VALUE,
                "log_diffusion": EMPTY_FIELD_VALUE,
            }

            for key in default_traj_info:
                if default_traj_info[key] is None:
                    continue
                if key not in traj_info:
                    traj_info[key] = default_traj_info[key]

            raw_positions -= raw_positions[0]
            time -= time[0]
            positions, clipped_steps = self.safe_positions(raw_positions, graph_info)

            edge_index = self.get_edges(positions.shape[0], graph_info)

            reshape = lambda t: torch.reshape(torch.from_numpy(t), (1, -1))
            float_to_torch = lambda t: torch.Tensor([t]).view((1, 1))

            # print("Skeleton seed = %d" % traj_info["seed"])

            super(SkeletonTrajData, self).__init__(
                pos=torch.from_numpy(positions).float(),
                original_pos=torch.from_numpy(original_positions).float(),
                time=torch.from_numpy(time).float(),
                clipped_steps=float_to_torch(clipped_steps),
                edge_index=edge_index,
                length=float_to_torch(positions.shape[0]),
                alpha=float_to_torch(traj_info["alpha"]),
                log_theta=float_to_torch(traj_info["log_theta"]),
                log_tau=reshape(np.asarray(np.log10(traj_info["tau"]))),
                model=float_to_torch(traj_info["model_index"]).long(),
                noise=float_to_torch(traj_info["pos_uncertainty"]),
                seed=torch.Tensor([int(traj_info["seed"])]),
                log_diffusion=float_to_torch(traj_info["log_diffusion"]),
            )
        except AttributeError as e:
            # print(e)
            super(SkeletonTrajData, self).__init__(
                pos=torch.from_numpy(np.random.normal(size=(10, 2))),
                time=torch.arange(10),
            )
        self.coalesce()

    def safe_positions(self, positions, graph_info):
        # clips too long jumps, returns the number of clipped steps
        if graph_info["clip_trajs"] == True:
            dr = get_steps(positions)
            M = np.median(dr) + 10 * np.std(dr)
            clipped_steps = np.sum(dr > M)
            dr_clipped = np.clip(dr, a_min=0.0, a_max=M)
            clipped_positions = np.zeros(positions.shape)
            deltas = positions[1:] - positions[:-1]
            dr[dr == 0.0] = 1.0  # in case
            ratio = dr_clipped / dr
            ratio[np.isnan(ratio)] = 0.0
            deltas = deltas * np.reshape(ratio, (-1, 1))
            clipped_positions[1:] = np.cumsum(deltas, axis=0)
            return clipped_positions, clipped_steps
        else:
            return positions, 0

    @classmethod
    def get_edges(cls, N, graph_info):

        D = graph_info["edges_per_point"]
        if D >= N:
            edge_start, edge_end = complete_graph(N)
        if graph_info["edge_method"] == "uniform":
            edge_start, edge_end = edges_uniform(N, D)
        elif graph_info["edge_method"] == "geom_causal":
            edge_start, edge_end = edges_geom_causal(N, D)
        else:
            raise NotImplementedError(f"Method { graph_info['edge_method'] } not known")
        e = np.stack([edge_start, edge_end], axis=0)
        return torch.from_numpy(e).long()
