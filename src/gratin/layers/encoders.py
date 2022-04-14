from torch._C import ThroughputBenchmark
from .diverse import *
from .minimal_conv import *
from torch_geometric.nn import NNConv, global_mean_pool, GCNConv
from torch_sparse import SparseTensor


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

        self.nodes_MLP = MLP([x_dim, 32, 32, x_dim])

        self.edges_MLP = MLP([e_dim, 32, 32, e_latent_dim])

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

        self.last_conv = MinimalJumpsConv(n_c, n_c, aggr="max")

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
