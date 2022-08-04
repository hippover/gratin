from .diverse import *
from .minimal_conv import *
from torch_geometric.nn import (
    global_mean_pool,
    GINConv,
    GINEConv,
    MessagePassing,
    GlobalAttention,
    GCNConv,
    NNConv,
    InstanceNorm,
)
from torch.nn import BatchNorm1d
from torch_sparse import SparseTensor
from torch_scatter import scatter


class TrajsEncoder2(nn.Module):
    def __init__(
        self,
        traj_dim: int,
        x_dim: int,
        e_dim: int,
        n_c: int = 64,  # Number of convolution kernels
        n_scales: int = 1,
        latent_dim: int = 8,
    ):
        super(TrajsEncoder2, self).__init__()
        e_latent_dim = 16

        self.nodes_MLP = MLP([x_dim, 128, x_dim], use_batch_norm=False)

        self.edges_MLP = MLP([e_dim, 128, e_latent_dim], use_batch_norm=False)

        self.att_conv = GCNConv(
            in_channels=x_dim,
            out_channels=n_c,
            improved=True,
        )

        self.conv_edges = NNConv(
            in_channels=n_c,
            out_channels=n_c,
            nn=MLP([e_latent_dim, e_latent_dim * 2, n_c * n_c]),
            aggr="mean",
        )

        self.last_conv = GCNConv(n_c, n_c, aggr="max", improved=True)

        gate_nn = MLP([3 * n_c, 256, 1])
        self.pooling = GlobalAttention(gate_nn=gate_nn)
        # self.pooling = global_mean_pool

        self.n_scales = n_scales
        self.traj_dim = traj_dim

        self.mlp = MLP(
            [3 * n_c + n_scales + traj_dim, 2 * latent_dim, latent_dim, latent_dim]
        )  # used to be tanh for last_activation

    def forward(self, data):
        adj_t = data.adj_t
        # print(data.adj_t[:, :, 0])
        row, col, edge_attr = adj_t.t().coo()
        sparse_adj_t = SparseTensor(col=col, row=row)
        edge_index = torch.stack([row, col], dim=0)

        x1 = self.att_conv(
            x=self.nodes_MLP(data.x),
            edge_index=sparse_adj_t,
        )

        edges_embedding = self.edges_MLP(edge_attr)

        # adj_t = adj_t.set_value(edges_embedding)

        # x2 = self.conv_edges(x=x1, edge_index=adj_t)
        x2 = self.conv_edges(x=x1, edge_index=edge_index, edge_attr=edges_embedding)
        x2 = torch.tanh(x2)

        x3 = self.last_conv(x=x2, edge_index=sparse_adj_t)

        x = torch.cat([x1, x2, x3], dim=1)
        # x = x1 + x2 + x3

        x = self.pooling(x=x, batch=data.batch)

        # print(x.size())

        if self.n_scales > 0:
            x = torch.cat((x, torch.log(data.scales + 1e-5)), dim=1)
        if self.traj_dim > 0:
            x = torch.cat((x, data.orientation), dim=1)

        # print(x.size())
        out = self.mlp(x)

        return out


class TrajsEncoder(nn.Module):
    def __init__(
        self,
        traj_dim: int,
        x_dim: int,
        e_dim: int,
        n_c: int = 64,  # Number of convolution kernels
        n_scales: int = 1,
        latent_dim: int = 8,
    ):
        super(TrajsEncoder, self).__init__()
        e_latent_dim = 8
        x_latent_dim = 8

        self.nodes_bn1 = InstanceNorm(in_channels=x_dim, momentum=0.99)
        self.nodes_bn2 = InstanceNorm(in_channels=n_c, momentum=0.5)
        self.nodes_bn3 = InstanceNorm(in_channels=n_c, momentum=0.5)
        self.nodes_bn4 = InstanceNorm(in_channels=n_c, momentum=0.5)
        self.nodes_bn5 = BatchNorm1d(3 * n_c, momentum=0.5)

        self.edges_bn_1 = InstanceNorm(in_channels=e_dim, momentum=0.99)
        self.edges_bn_2 = InstanceNorm(in_channels=e_latent_dim, momentum=0.5)

        self.nodes_MLP = MLP([x_dim, 32, 32, x_latent_dim])

        self.edges_MLP = MLP([e_dim, 32, 32, e_latent_dim])

        self.att_conv = GINConv(
            nn=MLP([x_latent_dim, 32, 32, n_c]),
        )

        self.conv_edges = GINEConv(
            nn=MLP([n_c, 32, 32, n_c]),
            edge_dim=e_latent_dim,
        )
        self.last_conv = GINEConv(
            nn=MLP([n_c, 32, 32, n_c]),
            edge_dim=e_latent_dim,
        )

        gate_nn = MLP([3 * n_c, 32, 1])
        # Si le pooling est une simple moyenne, les gradients transmis aux convolutions sur graphes sont très petits.
        # Je n'ai pas encore trouvé d'explication...
        self.pooling = GlobalAttention(gate_nn=gate_nn)
        # self.pooling = global_mean_pool

        self.n_scales = n_scales
        self.traj_dim = traj_dim

        self.mlp = MLP(
            [(3 * n_c) + n_scales, 32, latent_dim],
            use_batch_norm=True,
            use_weight_norm=False,
        )  # used to be tanh for last_activation

    def forward(self, data):
        # adj_t is sparse
        sparse_adj_t = data.adj_t
        i, j, edge_attr = sparse_adj_t.coo()
        # unsure which of i and j is the source_target.
        # But at the graph-level it should be OK (i and j are only used in batch[i])

        B = data.batch

        x0 = self.nodes_bn1(data.x, batch=B)
        x0 = self.nodes_MLP(x0)

        edge_attr = self.edges_bn_1(edge_attr, batch=B[i])
        edges_embedding = self.edges_MLP(edge_attr)
        edges_embedding = self.edges_bn_2(edges_embedding, batch=B[i])
        sparse_adj_t = sparse_adj_t.set_value(edges_embedding, layout="coo")

        assert data.x.shape[0] == x0.shape[0]
        x1 = self.att_conv(
            x=x0,
            edge_index=sparse_adj_t,
        )
        x1 = self.nodes_bn2(x1, batch=B)
        assert torch.sum(torch.isnan(x1)) == 0, print(x0, i, j)
        assert x1.shape[0] == x0.shape[0]

        x2 = self.conv_edges(x=x1, edge_index=sparse_adj_t)

        x2 = self.nodes_bn3(x2, batch=B)

        assert torch.sum(torch.isnan(x2)) == 0, x1

        x3 = self.last_conv(x=x2, edge_index=sparse_adj_t)
        x3 = self.nodes_bn4(x3, batch=B)

        assert torch.sum(torch.isnan(x3)) == 0, x2

        x = torch.cat([x1, x2, x3], dim=1)
        # x = x1 + x2 + x3

        # We cut trajectories in sub-trajectories
        # We first pool by sub-trajectories using attention
        # We then average over so-obtained vectors
        x = self.pooling(x=x, batch=B)
        x = self.nodes_bn5(x)

        if self.n_scales > 0:
            assert data.scales.shape[1] == self.n_scales
            x = torch.cat((x, torch.log(data.scales + 1e-5)), dim=1)

        out = self.mlp(x)

        assert out.shape[0] == data.alpha.shape[0]

        return out
