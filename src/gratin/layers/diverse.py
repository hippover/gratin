import torch.nn as nn
import torch
import numpy as np
from torch_geometric.nn import GlobalAttention, global_mean_pool
from ..data.data_tools import edges_geom_causal
from torch_sparse import SparseTensor
from torch_geometric.data import Batch
from torch.nn.utils import weight_norm

## Basic perceptron


class ExpLayer(nn.Module):
    def forward(self, x):
        return torch.exp(x)


class BoundSigmoid(nn.Module):
    def __init__(self, out_range):
        super(BoundSigmoid, self).__init__()
        self.out_range = out_range
        assert self.out_range[0] < self.out_range[1]

    def forward(self, x):
        return (
            torch.sigmoid(x) * (self.out_range[1] - self.out_range[0])
            + self.out_range[0]
        )


class ResidualNet(nn.Module):
    def __init__(self, mod):
        super(ResidualNet, self).__init__()
        self.mod = mod

    def forward(self, x):
        M = self.mod(x)
        return 0.1 * M + x[:, : M.shape[1]]


def MLP(
    channels,
    activation="leaky",
    last_activation="identity",
    use_batch_norm=False,  # used to be True. this is very unstable with True...
    bias=True,
    dropout=0.0,
    out_range=None,
    residual=False,
    use_weight_norm=True,  # False,
):

    if out_range is not None:
        last_activation = BoundSigmoid(out_range)
    elif last_activation == "identity":
        last_activation = nn.Identity()
    elif last_activation == "exponential":
        last_activation = ExpLayer()

    if activation == "leaky":
        activations = [nn.LeakyReLU(0.2) for i in range(1, len(channels) - 1)] + [
            last_activation
        ]
    elif activation == "ELU":
        activations = [nn.ELU() for i in range(1, len(channels) - 1)] + [
            last_activation
        ]
    else:
        activations = [nn.ReLU() for i in range(1, len(channels) - 1)] + [
            last_activation
        ]

    norm_function = lambda lin: lin
    if use_weight_norm:
        norm_function = lambda lin: weight_norm(lin)

    sequential = nn.Sequential(
        *[
            nn.Sequential(
                nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                norm_function(nn.Linear(channels[i - 1], channels[i], bias=bias)),
                nn.BatchNorm1d(channels[i]) if use_batch_norm else nn.Identity(),
                activations[i - 1],
            )
            for i in range(1, len(channels))
        ]
    )
    sequential[0].in_channels = channels[0]
    # sequential[-1].out_channels = channels[0]
    sequential[-1].out_channels = channels[-1]

    if not residual:
        return sequential
    else:
        return ResidualNet(sequential)


def batch_from_positions(pos, N, L, D, degree):
    x_pos = torch.reshape(pos, (N * L, D))
    # assert torch.equal(x_pos[L : (2 * L)], pos[1])
    batch = torch.repeat_interleave(torch.arange(N, device=pos.device), L)
    # edge_index = x.get_edges(steps[0], {"edge_method": "geom_causal"})
    row, col = edges_geom_causal(L, degree)
    row = torch.from_numpy(row).to(x_pos.device)
    col = torch.from_numpy(col).to(x_pos.device)
    edge_index = torch.stack((row, col), dim=0).long()

    N_edges = edge_index.shape[1]
    edge_index = edge_index.repeat((1, N))
    shift = (
        torch.repeat_interleave(torch.arange(N, device=x_pos.device).long(), N_edges)
        * L
    )
    edge_index += torch.stack((shift, shift), dim=0)
    # adj_t = SparseTensor(col=edge_index[0], row=edge_index[1])

    x_pred = Batch(batch=batch, pos=x_pos, edge_index=edge_index)

    return x_pred


def batch_from_sub_batches(sub_batches):
    pos_vectors = []
    edge_index_offset = 0
    batch_offset = 0
    batch_vectors = []
    nodes_offset = 0
    edge_indices = []

    assert "edge_index" in sub_batches[0].keys

    other_keys = list(
        set(sub_batches[0].keys) - set(["pos", "ptr", "edge_index", "batch"])
    )
    other_keys_dict = {}

    for k in other_keys:
        other_keys_dict[k] = []

    for batch in sub_batches:
        B = batch.batch + batch_offset
        batch_vectors.append(B)
        batch_offset = torch.max(B) + 1

        pos_vectors.append(batch.pos)

        e = batch.edge_index + nodes_offset
        edge_indices.append(e)
        nodes_offset += batch.pos.shape[0]

        for k in other_keys:
            other_keys_dict[k].append(batch[k])

    edge_index = torch.cat(edge_indices, dim=1)
    pos = torch.cat(pos_vectors, dim=0)
    batch = torch.cat(batch_vectors, dim=0)

    assert torch.max(edge_index) == pos.shape[0] - 1
    assert torch.min(edge_index[1:, 1] - edge_index[:-1, 1]) >= 0

    adj_t = SparseTensor(col=edge_index[0], row=edge_index[1])
    new_batch = Batch(batch=batch, pos=pos, adj_t=adj_t)

    for k in other_keys:
        try:
            new_batch[k] = torch.cat(other_keys_dict[k], dim=0)
        except Exception as e:
            print("key = %s" % k)
            raise e

    return new_batch


def generate_batch_like(
    x,
    T_values,
    alpha_range,
    tau_range,
    generator,
    degree,
    simulate_tau=False,
    simulate_diffusion=True,
):
    batches = []
    BS = torch.max(x.batch) + 1
    # print("BS = %d" % BS)
    SBS_min = BS // len(T_values)
    # print("SBS = %s" % SBS)

    # if not x.pos.is_cuda:
    #    x.pos = x.pos.cuda()

    for T in T_values:
        SBS = int(SBS_min)
        # print("Generate %d trajs of length %d" % (SBS, T))
        # print("T = %d, generating %d trajs" % (T, SBS))
        # ALPHA
        alpha = (
            torch.rand(SBS, device=x.pos.device) * (alpha_range[1] - alpha_range[0])
            + alpha_range[0]
        )
        # print("alpha = ", alpha)

        # TAU if needed
        if simulate_tau:
            log_tau = torch.rand(SBS, device=x.pos.device) * (
                np.log10(tau_range[1]) - np.log10(tau_range[0])
            ) + np.log10(tau_range[0])
        else:
            log_tau = torch.ones_like(alpha) * np.log10(T) + 1
        tau = torch.floor(torch.pow(10.0, log_tau))
        log_tau = torch.log10(tau)

        # print("tau = ", tau)
        # Make sure that tau is not larger than T
        # tau = torch.where(tau > T, T, tau)

        # DIFFUSION
        if simulate_diffusion == True:
            log_diffusion = torch.rand(SBS, device=x.pos.device) * 4 - 2
        else:
            log_diffusion = torch.ones_like(alpha) * 0.0
        diffusion = torch.pow(10.0, log_diffusion)

        pos = generator(alpha, tau, diffusion, T)
        # print("Generating T = %d" % T)
        # print(pos.shape)

        x = batch_from_positions(
            pos,
            N=SBS,
            L=T,
            D=generator.dim,
            degree=degree,
        )
        x.alpha = alpha.view(-1, 1)
        x.log_tau = log_tau.view(-1, 1)
        x.log_diffusion = log_diffusion.view(-1, 1)
        x.length = torch.ones_like(x.alpha) * T
        assert x.batch.shape[0] == SBS * T
        batches.append(x)

    # print("num of sub_batches %d " % len(batches))

    x = batch_from_sub_batches(batches)

    return x


class AlphaPredictor(nn.Module):
    """
    The only interest of this class is to add an offset and to restrict the output to a plausible interval
    """

    def __init__(
        self,
        p=0.0,
        input_dim=128,
        mlp_size=[128, 128, 64, 16],
    ):

        super(AlphaPredictor, self).__init__()
        MLP_size = [input_dim] + mlp_size + [1]
        self.mlp = MLP(MLP_size, dropout=p)

    def forward(self, x):
        residual = self.mlp(x)
        return 1.0 + 0.99 * torch.nn.Tanh()(residual)
