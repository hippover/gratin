import torch.nn as nn
import torch
import numpy as np
from torch_sparse import matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import GlobalAttention, global_mean_pool
from ..data.data_tools import edges_geom_causal
from torch_sparse import SparseTensor
from torch_geometric.data import Batch

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


def MLP(
    channels,
    activation="leaky",
    last_activation="identity",
    bias=True,
    dropout=0.0,
    out_range=None,
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
    else:
        activations = [nn.ReLU() for i in range(1, len(channels) - 1)] + [
            last_activation
        ]
    return nn.Sequential(
        *[
            nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(channels[i - 1], channels[i], bias=bias),
                nn.BatchNorm1d(channels[i]),
                activations[i - 1],
            )
            for i in range(1, len(channels))
        ]
    )


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
    adj_t = SparseTensor(col=edge_index[0], row=edge_index[1])

    x_pred = Batch(batch=batch, pos=x_pos, adj_t=adj_t)
    return x_pred


class AlphaPredictor(nn.Module):
    """
    The only interest of this class is to add an offset and to restrict the output to a plausible interval
    """

    def __init__(
        self,
        p=0.0,
        input_dim=128,
        alpha_fit=False,
        subdiffusive_only=False,
        mlp_size=[128, 128, 64, 16],
    ):
        """
        alpha_fit : whether the latent space has its last dimension indicating TAMSD fit
        """
        super(AlphaPredictor, self).__init__()
        if alpha_fit:
            self.bn_alpha_fit = nn.BatchNorm1d(1)
        self.alpha_fit = alpha_fit
        self.subdiffusive_only = subdiffusive_only
        MLP_size = [input_dim] + mlp_size + [1]
        self.mlp = nn.Sequential(MLP(MLP_size, dropout=p))

        nparams = 0
        for n, p in self.named_parameters():
            np_ = np.product(np.array([s for s in p.shape]))
            # print(n, np_, p.shape)
            nparams += np_
        print("alpha MLP size = ", MLP_size)
        print("Alpha predictor has %d parameters" % nparams)

    def forward(self, x):
        # We only normalize the last column (alpha_fit)
        # as all others have already been batch-normalized at this stage
        if self.alpha_fit:
            x[:, -1:] = self.bn_alpha_fit(x[:, -1:])
        residual = self.mlp(x)  # Last column of x is the alpha fit by MSD
        if self.subdiffusive_only:
            return 0.5 + 0.5 * torch.nn.Tanh()(residual)
        else:
            return 1.0 + 0.99 * torch.nn.Tanh()(residual)
