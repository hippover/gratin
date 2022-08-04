from platform import node
from torch.functional import cdist
import torch.nn as nn
from torch_geometric.data import Batch, Data
from torch_geometric.utils.subgraph import subgraph
from torch_scatter import scatter
import torch


def get_graph_indices(X: Data):
    # source, target, _ = X.adj_t.t().coo()
    target, source, _ = X.adj_t.coo()
    return source, target


def scatter_std(X, B):
    avg_X_2 = scatter(src=X**2, index=B, dim=0, reduce="mean")
    avg_X = scatter(src=X, index=B, dim=0, reduce="mean")
    Var = avg_X_2 - (avg_X**2)
    return torch.sqrt(Var)


def is_first_point(B):
    # print(B)
    f = torch.abs(B - torch.roll(B, 1, dims=(0,))) > 0
    f = f | (torch.arange(B.shape[0], device=B.device) == 0)
    # print("first points", f)
    assert f[0] == True
    assert torch.sum(f) == torch.unique(B).shape[0]
    return f


def is_last_point(B):
    # print(B)
    f = torch.abs(B - torch.roll(B, -1, dims=(0,))) > 0
    f[-1] = True
    # print("last points", f)
    if torch.max(B) > torch.min(B):
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


def cumsum_per_graph(X, B, step_mode=False):
    assert len(X.shape) == 1, "Does not work well when more than 1 dim"
    # Supposed to work on steps, where item i contains jump size from i to i + 1
    # first value of returned array is 0
    cumsum = torch.cumsum(X, dim=0)
    first_point = 1 * is_first_point(B)
    first_cumsum_b = scatter(cumsum * first_point, B, dim=0, reduce="sum")
    first_value = scatter(X * first_point, B, dim=0, reduce="sum")

    # last_point = 1 * is_last_point(B)
    # last_value = scatter(cumsum * last_point, B, dim=0, reduce="sum")

    # if not step_mode:
    cumsum = cumsum - torch.index_select(input=first_cumsum_b, dim=0, index=B)
    cumsum = cumsum + torch.index_select(input=first_value, dim=0, index=B)
    if step_mode:
        cumsum = cumsum.roll(1, dims=0)
        cumsum[is_first_point(B)] = 0.0
    # cumsum = cumsum - torch.index_select(input=first_cumsum_b, dim=0, index=B)
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
    def get_scales(self, B, P):
        scales = {}
        dr = diff_per_graph(P, B)
        in_points = ~is_last_point(B)
        dr = dr[in_points]
        dr_norm = torch.sqrt(1e-5 + torch.sum(dr**2, dim=1))

        P_STD = torch.sqrt(torch.sum(scatter_std(P, B) ** 2, dim=1))
        scales["pos_std"] = P_STD

        step_sum = scatter(dr_norm, index=B[in_points], dim=0, reduce="sum")
        scales["step_sum"] = step_sum

        step_std = scatter_std(dr_norm, B[in_points])
        scales["step_std"] = step_std

        step_mean = scatter(dr_norm, index=B[in_points], dim=0, reduce="mean")
        scales["step_mean"] = step_mean

        # var = <dr^2> - <dr>^2
        step_var = scatter(dr_norm**2, index=B[in_points], dim=0, reduce="mean")
        # - torch.sum(
        #    scatter(dr, index=B[in_points], dim=0, reduce="mean") ** 2, dim=1
        # )
        scales["step_var"] = step_var

        return scales

    @classmethod
    def x_dim(cls, scale_types):
        return 1 + len(scale_types) * 2

    @classmethod
    def e_dim(cls, scale_types):
        return 1 + len(scale_types) * 4

    def forward(self, data: Batch, scale_types=["pos_std"], return_intermediate=False):
        B = data.batch
        P = data.pos

        source, target = get_graph_indices(data)
        scales = self.get_scales(B, P, scale_types)

        time = cumsum_per_graph(torch.ones_like(B), B)
        L = scatter(torch.ones_like(B), B, dim=0, reduce="sum")
        time_norm = (time - 1) / (torch.index_select(L, dim=0, index=B) - 1)
        assert torch.min(scatter(time_norm, B, reduce="max")) == 1.0, time_norm
        assert torch.max(scatter(time_norm, B, reduce="min")) == 0.0, time_norm
        node_features = [time_norm.view(-1, 1).float()]
        time_delta = (time_norm[target] - time_norm[source]).view(-1, 1).float()
        edge_features = [time_delta]

        dr = diff_per_graph(P, B)

        for k in scales:
            s = scales[k]
            scale_factor = torch.index_select(s, dim=0, index=B).view(-1, 1)
            dr_ = dr / scale_factor
            p = P / scale_factor
            dr_norm = torch.sqrt(1e-5 + torch.sum(dr_**2, dim=1))

            cum_dist = cumsum_per_graph(dr_norm, B).view(
                -1,
            )
            cum_sd = cumsum_per_graph(dr_norm**2, B).view(
                -1,
            )
            total_dist = scatter(dr_norm, B, dim=0, reduce="sum")
            total_dist = torch.index_select(total_dist, dim=0, index=B).view(
                -1,
            )
            total_sd = scatter(dr_norm**2, B, dim=0, reduce="sum")
            total_sd = torch.index_select(total_sd, dim=0, index=B).view(
                -1,
            )

            assert cum_dist.shape[0] == total_dist.shape[0]
            cum_dist = (
                cum_dist
                / time_norm.view(
                    -1,
                )
            ).view(-1, 1)
            cum_dist[time_norm == 0] = 0.0
            cum_sd = (
                cum_sd
                / time_norm.view(
                    -1,
                )
            ).view(-1, 1)
            cum_sd[time_norm == 0] = 0.0
            node_features.append(cum_dist)  # there was no / time
            node_features.append(cum_sd)  # there was no / time
            # node_features.append(cummax_per_graph(dr_norm, B).view(-1, 1))

            end, start = p[target], p[source]
            d = end - start
            d = torch.sqrt(torch.sum(d**2, dim=1))

            L_ = torch.index_select(L, dim=0, index=B).view(-1, 1)
            end_jump, start_jump = dr_[target] * L_[target], dr_[source] * L_[source]
            corr = torch.sum(end_jump * start_jump, dim=1)

            edge_features.append(
                (
                    d.view(
                        -1,
                    )
                    / torch.abs(time_delta).view(
                        -1,
                    )
                ).view(-1, 1)
            )
            edge_features.append(corr.view(-1, 1))
            edge_features.append(
                (
                    (cum_dist[target] - cum_dist[source]).view(
                        -1,
                    )
                    / torch.abs(time_delta).view(
                        -1,
                    )
                ).view(-1, 1)
            )
            edge_features.append(
                (
                    (cum_sd[target] - cum_sd[source]).view(
                        -1,
                    )
                    / torch.abs(time_delta).view(
                        -1,
                    )
                ).view(-1, 1)
            )
            assert torch.allclose(total_dist[target], total_dist[source])
        X = torch.cat(node_features, dim=1)
        E = torch.cat(edge_features, dim=1)

        assert X.shape[1] == TrajsFeatures.x_dim(scale_types), "%d vs %d" % (
            X.shape[1],
            TrajsFeatures.x_dim(scale_types),
        )
        assert E.shape[1] == TrajsFeatures.e_dim(scale_types)
        assert torch.sum(torch.isnan(X)) == 0, X
        assert torch.sum(torch.isnan(E)) == 0, E

        scales["log_L"] = torch.log(L.float())

        return X, E, scales


class TrajsFeaturesSimple(TrajsFeatures):
    @classmethod
    def x_dim(cls):
        return 6

    @classmethod
    def e_dim(cls):
        return 6

    def forward(self, data: Batch, return_intermediate=False):
        B = data.batch
        P = data.pos
        time = data.time

        source, target = get_graph_indices(data)
        scales = self.get_scales(B, P)

        n_points = scatter(torch.ones_like(B), B, reduce="sum")
        duration = scatter(time, index=B, dim=0, reduce="max")
        # print(n_points)
        # print(duration)

        scales["mean_time_step"] = duration / n_points

        start = scatter(time, index=B, dim=0, reduce="min")
        assert torch.max(start) <= 0
        time_norm = time / torch.index_select(duration, dim=0, index=B)
        assert torch.min(scatter(time_norm, B, reduce="max")) == 1.0, time_norm
        assert torch.max(scatter(time_norm, B, reduce="min")) == 0.0, time_norm
        node_features = [time_norm.view(-1, 1).float()]
        time_delta = (time[target] - time[source]).view(-1).float()
        time_delta_norm = (time_norm[target] - time_norm[source]).view(-1).float()
        edge_features = [time_delta]

        dr = diff_per_graph(P, B)

        dr_norm = torch.sqrt(1e-5 + torch.sum(dr**2, dim=1))
        cum_dist = cumsum_per_graph(dr_norm, B).view(
            -1,
        )
        cum_sd = cumsum_per_graph(dr_norm**2, B).view(
            -1,
        )
        cum_qd = cumsum_per_graph(dr_norm**4, B).view(
            -1,
        )

        dist_to_origin = torch.sqrt(torch.sum(P**2, dim=1) + 1e-7)
        max_dist_to_origin = cummax_per_graph(dist_to_origin, B)

        total_dist = scatter(dr_norm, B, dim=0, reduce="sum")
        total_dist = torch.index_select(total_dist, dim=0, index=B).view(-1)
        total_sd = scatter(dr_norm**2, B, dim=0, reduce="sum")
        total_sd = torch.index_select(total_sd, dim=0, index=B).view(-1)
        total_qd = scatter(dr_norm**4, B, dim=0, reduce="sum")
        total_qd = torch.index_select(total_qd, dim=0, index=B).view(-1)

        end, start = P[target], P[source]
        d = end - start
        d = torch.sqrt(torch.sum(d**2, dim=1))

        step_std = torch.index_select(scales["step_std"], dim=0, index=B).view(-1)
        end_jump, start_jump = dr[target], dr[source]
        corr = torch.sum(end_jump * start_jump, dim=1) / (
            step_std[target] * step_std[source]
        )

        node_features.append(cum_dist / (total_dist * time_norm))
        node_features.append(cum_sd / (total_sd * time_norm))
        node_features.append(cum_qd / (total_qd * time_norm))
        node_features.append(dist_to_origin / (step_std * time.sqrt()))
        node_features.append(max_dist_to_origin / (step_std * time.sqrt()))

        edge_features.append(
            cum_dist[target]
            - cum_dist[source] / (total_dist[target] * time_delta_norm.abs())
        )
        edge_features.append(
            cum_sd[target] - cum_sd[source] / (total_sd[target] * time_delta_norm.abs())
        )
        edge_features.append(
            cum_qd[target] - cum_qd[source] / (total_qd[target] * time_delta_norm.abs())
        )
        edge_features.append(d / (step_std[target] * time_delta.abs().sqrt()))
        edge_features.append(corr)

        for i, x in enumerate(node_features):
            x[time_norm == 0] = 0.0
            node_features[i] = x.view(-1, 1)

        for i, e in enumerate(edge_features):
            e[time_delta == 0] = 0.0
            edge_features[i] = e.view(-1, 1)

        X = torch.cat(node_features, dim=1)
        E = torch.cat(edge_features, dim=1)

        assert X.shape[1] == TrajsFeaturesSimple.x_dim(), "%d vs %d" % (
            X.shape[1],
            TrajsFeaturesSimple.x_dim(),
        )
        assert E.shape[1] == TrajsFeaturesSimple.e_dim()
        assert torch.sum(torch.isnan(X)) == 0, X
        assert torch.sum(torch.isnan(E)) == 0, E

        return X, E, scales


class TrajsNoFeatures(TrajsFeatures):
    @classmethod
    def x_dim(cls):
        return 2

    @classmethod
    def e_dim(cls):
        return 2

    def forward(self, data: Batch, return_intermediate=False):
        # B = data.batch
        B = data.trajectory_batch
        P = data.pos

        source, target = get_graph_indices(data)
        scales = self.get_scales(B, P)
        # col > row
        assert torch.min(target - source) > 0, (
            target,
            source,
            torch.min(target - source),
        )
        time = cumsum_per_graph(torch.ones_like(B), B, step_mode=True)
        time = time.view(-1).float()
        L = scatter(torch.ones_like(B), B, dim=0, reduce="sum")
        L_10 = torch.div(L, 8, rounding_mode="trunc")
        L_10_trajs = torch.index_select(L_10, dim=0, index=B)
        include_in_10 = time % L_10_trajs == 0
        pos_10 = P[include_in_10]
        dr_10, B_10 = diff_per_graph(pos_10, B[include_in_10], fill_last=False)
        scale_10 = scatter(dr_10**2, index=B_10, dim=0, reduce="mean").sqrt().view(-1)

        time_norm = time / (torch.index_select(L, dim=0, index=B) - 1)
        # assert torch.min(scatter(time_norm, B, reduce="max")) == 1.0, time_norm
        # assert torch.max(scatter(time_norm, B, reduce="min")) == 0.0, time_norm
        node_features = [time_norm.view(-1, 1).float()]
        time_delta = (time[target] - time[source]).view(-1).float()
        assert time_delta.min() > 0, time_delta.min()  # row > col
        time_delta_norm = (time_norm[target] - time_norm[source]).view(-1).float()
        edge_features = []
        # edge_features.append(time_delta_norm)

        dist_to_origin = torch.sqrt(torch.sum(P**2, dim=1) + 1e-7)
        max_dist_to_origin = cummax_per_graph(dist_to_origin, B)

        end, start = P[target], P[source]

        d = end - start
        d = torch.sqrt(torch.sum(d**2, dim=1))

        step_std = torch.index_select(scales["step_std"], dim=0, index=B).view(-1)
        node_features.append(dist_to_origin / (step_std * time.sqrt()))
        # node_features.append(
        #    (dist_to_origin / torch.index_select(scale_10, dim=0, index=B)) ** 2
        # )
        edge_features.append(
            time_delta_norm
        )  # mean across nodes is invariant to length
        # edge_features.append(time_delta_norm)
        # edge_features.append(
        #    torch.log(
        #        torch.clamp(d**2, min=1e-6, max=torch.inf)
        #        / (max_dist_to_origin[row] ** 2)
        #    )
        # )
        # edge_features.append(
        #    (d / torch.index_select(scale_10, dim=0, index=B[row])) ** 2
        #    / time_delta.abs()
        # )

        # worked well ! with norm time delta
        edge_features.append(d / (step_std[target] * (time_delta.abs().sqrt())))

        # mean_d = scatter(d**2, B[row], reduce="mean").sqrt()
        # mean_dist_to_o = scatter(dist_to_origin**2, B, reduce="mean").sqrt()
        # node_features.append(
        #    dist_to_origin / torch.index_select(scale_10, dim=0, index=B)
        # )
        # edge_features.append(
        #    d**2 / torch.index_select(scale_10**2, dim=0, index=B[row])
        # )

        for i, x in enumerate(node_features):
            x[time_norm == 0] = 0.0
            node_features[i] = x.view(-1, 1)

        for i, e in enumerate(edge_features):
            # e[time_delta == 0] = 0.0
            e[time[source] == 0] = 1.0
            edge_features[i] = e.view(-1, 1)

        X = torch.cat(node_features, dim=1)
        E = torch.cat(edge_features, dim=1)

        assert X.shape[1] == self.x_dim(), "%d vs %d" % (
            X.shape[1],
            TrajsFeaturesSimple.x_dim(),
        )
        assert E.shape[1] == self.e_dim()
        assert torch.sum(torch.isnan(X)) == 0, X
        assert torch.sum(torch.isnan(E)) == 0, E
        assert torch.sum(torch.isinf(X)) == 0, X
        assert torch.sum(torch.isinf(E)) == 0, E

        scales["log_L"] = torch.log(L.float())

        return X, E, scales
