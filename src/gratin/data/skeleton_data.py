import numpy as np
from ..simulation.traj_tools import *
from torch_geometric.data import Data
import torch
import numpy as np

from .data_tools import *

EMPTY_FIELD_VALUE = -999


class SkeletonTrajData(Data):
    def __init__(
        self,
        raw_positions,
        graph_info={},
        traj_info={},
        original_positions=None,
        **kwargs,
    ):

        # if raw_positions is None:
        #    print("raw_positions was none")
        #    raw_positions = np.random.normal(size=(10, 2))
        try:
            dim = raw_positions.shape[1]

            if original_positions is None:
                original_positions = raw_positions
            if not type(original_positions) is np.array:
                original_positions = np.array(original_positions).copy()

            default_traj_info = {
                "model": "unknown",
                "model_index": EMPTY_FIELD_VALUE,
                "drift_norm": EMPTY_FIELD_VALUE,
                "drift_vec": np.ones(dim) * EMPTY_FIELD_VALUE,
                "force_norm": EMPTY_FIELD_VALUE,
                "force_vec": np.ones(dim) * EMPTY_FIELD_VALUE,
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
            positions, clipped_steps = self.safe_positions(raw_positions, graph_info)

            X = self.get_node_features(positions, graph_info)
            edge_index = self.get_edges(X, graph_info)
            E = self.get_edge_features(X, positions, edge_index, graph_info)

            reshape = lambda t: torch.reshape(torch.from_numpy(t), (1, -1))
            float_to_torch = lambda t: torch.Tensor([t]).view((1, 1))

            # print("Skeleton seed = %d" % traj_info["seed"])

            super(SkeletonTrajData, self).__init__(
                pos=torch.from_numpy(positions).float(),
                x=torch.from_numpy(X).float(),
                original_pos=torch.from_numpy(original_positions).float(),
                clipped_steps=float_to_torch(clipped_steps),
                edge_index=edge_index,
                edge_attr=torch.from_numpy(E).float() if E.shape[1] > 0 else None,
                length=float_to_torch(positions.shape[0]),
                alpha=float_to_torch(traj_info["alpha"]),
                log_theta=float_to_torch(traj_info["log_theta"]),
                drift_norm=float_to_torch(traj_info["drift_norm"]),
                drift=reshape(traj_info["drift_vec"]),
                force_norm=float_to_torch(traj_info["force_norm"]),
                force=reshape(traj_info["force_vec"]),
                log_tau=reshape(np.asarray(np.log10(traj_info["tau"]))),
                model=float_to_torch(traj_info["model_index"]).long(),
                noise=float_to_torch(traj_info["pos_uncertainty"]),
                seed=torch.Tensor([int(traj_info["seed"])]),
                log_diffusion=float_to_torch(traj_info["log_diffusion"]),
            )
        except AttributeError as e:
            # print(e)
            super(SkeletonTrajData, self).__init__(
                pos=torch.from_numpy(np.random.normal(size=(10, 2)))
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
    def get_edges(cls, X, graph_info):

        D = graph_info["edges_per_point"]
        N = X.shape[0]
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

    def get_node_features(self, positions, graph_info):
        # Minimal feature : time
        return np.reshape(np.arange(positions.shape[0]), (-1, 1)) / positions.shape[0]

    def get_edge_features(self, X, positions, edge_index, graph_info):
        if graph_info["features_on_edges"] == False:
            return np.zeros((edge_index.shape[1], 0))
        else:
            raise NotImplementedError("Ne sait pas calculer les features sur les edges")
        return None
