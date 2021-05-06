from typing import Union
import torch.nn as nn
from torch_geometric.data import Batch
from torch_scatter import scatter
import torch


def scatter_std(X, B):
    avg_X_2 = scatter(src=X ** 2, index=B, dim=0, reduce="mean")
    avg_X = scatter(src=X, index=B, dim=0, reduce="mean")
    Var = avg_X_2 - (avg_X ** 2)
    return torch.sqrt(Var)


def is_first_point(B):
    # print(B)
    f = torch.abs(B - torch.roll(B, 1, dims=(0,))) > 0
    # print("first points", f)
    assert f[0] == True
    return f


def is_last_point(B):
    # print(B)
    f = torch.abs(B - torch.roll(B, -1, dims=(0,))) > 0
    # print("last points", f)
    assert f[-1] == True
    return f


def diff_per_graph(X, B, fill_last=True):
    d = torch.roll(X, -1, dims=(0,)) - X
    if fill_last:
        # d[is_last_point(B)] = 0.0
        return torch.where(
            torch.stack([is_last_point(B) for d in range(X.shape[1])], dim=1),
            0.0,
            d.double(),
        ).float()
    else:
        d = d[~is_last_point(B)]
        return d, B[~is_last_point(B)]


def cumsum_per_graph(X, B):
    cumsum = torch.cumsum(X, dim=0)
    min_per_b = scatter(cumsum, B, dim=0, reduce="min")

    first_point = 1 * is_first_point(B)
    first_value = scatter(X * first_point, B, dim=0, reduce="sum")

    cumsum = cumsum - torch.index_select(input=min_per_b, dim=0, index=B)
    cumsum = cumsum + torch.index_select(input=first_value, dim=0, index=B)
    return cumsum


def cummax_per_graph(X, B):
    cummax, _ = torch.cummax(X, dim=0)
    min_per_b = scatter(cummax, B, dim=0, reduce="min")

    first_point = 1 * is_first_point(B)
    first_value = scatter(X * first_point, B, dim=0, reduce="sum")

    cumsum = cummax - torch.index_select(input=min_per_b, dim=0, index=B)
    cumsum = cummax + torch.index_select(input=first_value, dim=0, index=B)
    return cumsum


class TrajsFeatures(nn.Module):
    def get_scales(self, B, P, scale_types):
        scales = []

        if scale_types != ["pos_std"]:
            dr = diff_per_graph(P, B)
            in_points = ~is_first_point(B)
            dr = dr[in_points]
            dr_norm = torch.sqrt(1e-5 + torch.sum(dr ** 2, dim=1))

        if "pos_std" in scale_types:
            P_STD = torch.sqrt(torch.sum(scatter_std(P, B) ** 2, dim=1))
            scales.append(P_STD)

        if "step_sum" in scale_types:
            step_sum = scatter(dr_norm, index=B[in_points], dim=0, reduce="sum")
            scales.append(step_sum)

        if "step_std" in scale_types:
            step_std = scatter_std(dr_norm, B[in_points])
            scales.append(step_std)

        if "step_mean" in scale_types:
            step_mean = scatter(dr_norm, index=B[in_points], dim=0, reduce="mean")
            scales.append(step_mean)
        return scales

    @classmethod
    def x_dim(cls, scale_types):
        return 1 + len(scale_types) * 3

    @classmethod
    def e_dim(cls, scale_types):
        return 1 + len(scale_types) * 4

    def forward(self, data: Batch, scale_types=["pos_std"], return_intermediate=False):
        B = data.batch
        P = data.pos

        row, col, _ = data.adj_t.t().coo()
        scales = self.get_scales(B, P, scale_types)

        time = cumsum_per_graph(torch.ones_like(B), B)
        L = scatter(torch.ones_like(B), B, dim=0, reduce="sum")
        time_norm = time / torch.index_select(L, dim=0, index=B)

        node_features = [time_norm.view(-1, 1).float()]
        edge_features = [(time_norm[row] - time_norm[col]).view(-1, 1).float()]

        dr = diff_per_graph(P, B)

        for s in scales:

            scale_factor = torch.index_select(s, dim=0, index=B).view(-1, 1)
            dr_ = dr / scale_factor
            p = P / scale_factor
            dr_norm = torch.sqrt(1e-5 + torch.sum(dr_ ** 2, dim=1))

            cum_dist = cumsum_per_graph(dr_norm, B).view(-1, 1)
            cum_msd = cumsum_per_graph(dr_norm ** 2, B).view(-1, 1)
            node_features.append(cum_dist)
            node_features.append(cum_msd)
            node_features.append(cummax_per_graph(dr_norm, B).view(-1, 1))

            end, start = p[row], p[col]
            d = end - start
            d = torch.sqrt(torch.sum(d ** 2, dim=1))

            end_jump, start_jump = dr_[row], dr_[col]
            corr = torch.sum(end_jump * start_jump, dim=1)

            edge_features.append(d.view(-1, 1))
            edge_features.append(corr.view(-1, 1))
            edge_features.append(cum_dist[row] - cum_dist[col])
            edge_features.append(cum_msd[row] - cum_msd[col])

        X = torch.cat(node_features, dim=1)
        E = torch.cat(edge_features, dim=1)

        assert X.shape[1] == TrajsFeatures.x_dim(scale_types)
        assert E.shape[1] == TrajsFeatures.e_dim(scale_types)

        # Make a tensor with scales and L (to be used to infer D)
        u = scatter(dr, B, reduce="mean", dim=0)
        u = u / torch.sqrt(1e-5 + torch.sum(u ** 2, dim=1).view(-1, 1))
        scales = torch.stack(scales + [L], dim=1)

        orientation = u

        if not return_intermediate:
            return X, E, scales, orientation
        else:
            return X, E, scales, orientation, node_features, edge_features, dr_norm
