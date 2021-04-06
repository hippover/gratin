from .diverse import *
from .minimal_conv import *
from torch_geometric.nn import NNConv, GATConv


class TrajsEncoder2(nn.Module):
    def __init__(
        self,
        x_dim: int,
        e_dim: int,
        n_c: int = 64,  # Number of convolution kernels
        latent_dim: int = 8,
    ):
        super(TrajsEncoder2, self).__init__()

        self.edges_MLP = MLP([e_dim, 8, 8, 16])

        self.att_conv = GATConv(in_channels=x_dim, out_channels=n_c, heads=1)

        self.conv_edges = NNConv(
            in_channels=n_c,
            out_channels=n_c,
            nn=MLP([e_dim, e_dim * 2, x_dim * n_c]),
            aggr="mean",
        )

        self.last_conv = MinimalJumpsConv(latent_dim, 2 * n_c)

        gate_nn = MLP([3 * n_c, n_c, n_c // 2, 1])
        self.pooling = GlobalAttention(gate_nn=gate_nn)

        self.mlp = MLP([3 * n_c, latent_dim])  # used to be tanh for last_activation

    def forward(self, data):
        adj_t = data.adj_t
        x1 = self.att_conv(x=data.x, edge_index=adj_t)
        adj_t = self.edges_MLP(adj_t)

        x2 = self.conv_edges(x=x1, edge_index=adj_t)

        x3 = self.last_conv(x=x2, edge_index=adj_t)

        x = torch.cat([x1, x2, x3], dim=1)
        x = self.pooling(x=x, batch=data.batch)
        return self.mlp(x)


## Encoder module
class TrajsEncoder(nn.Module):
    """
    Without edge features
    Succession of minimal jump conv
    Followed by a pooling layer
    And a final MLP that projects to the latent space
    """

    def __init__(
        self,
        x_dim,
        n_c: int = 64,  # Number of convolution kernels
        latent_dim: int = 8,
    ):
        super(TrajsEncoder, self).__init__()
        # To compute moments of features

        Conv = MinimalJumpsConv

        f_inner_width = [128, 64]
        moments = [1]
        n_final_convolutions = 1

        self.conv1 = Conv(
            out_channels=n_c,
            x_dim=x_dim,
            f_inner_width=f_inner_width,
            aggr="mean",
            moments=moments,
        )

        self.conv2 = Conv(
            out_channels=n_c,
            x_dim=n_c,
            f_inner_width=f_inner_width,
            aggr="max",
        )
        final_convs = []
        for i in range(n_final_convolutions):
            final_convs.append(
                Conv(
                    out_channels=n_c,
                    x_dim=(1 + 1 * (i == 0)) * n_c,
                    f_inner_width=f_inner_width,
                    aggr="mean",
                )
            )
        self.final_convs = nn.ModuleList(final_convs)

        K = 2 + n_final_convolutions
        # K = 1
        # if params_scarcity == 0:
        gate_nn = MLP([K * n_c, n_c, n_c // 2, 1])
        self.pooling = GlobalAttention(gate_nn=gate_nn)

        self.mlp = MLP([K * n_c, latent_dim])  # used to be tanh for last_activation
        # self.float()
        # print("Final projector has size ", mlp_size)

    def call_conv(self, conv, x, edge_index):
        return conv(x=x, edge_index=edge_index)

    def forward(self, data):

        edge_index = data.adj_t  # .cuda()

        x_1 = self.call_conv(self.conv1, x=data.x, edge_index=edge_index)
        x_2 = self.call_conv(self.conv2, x=x_1, edge_index=edge_index)

        # Concat two first depths of convolutions
        x = torch.cat([x_1, x_2], dim=1)

        convolved = [x]
        for _, conv in enumerate(self.final_convs):
            convolved.append(
                self.call_conv(conv, x=convolved[-1], edge_index=edge_index)
            )
        # Concat all depths of convolutions
        x = torch.cat(convolved, dim=1)
        # "average" with attention over all nodes
        x = self.pooling(x=x, batch=data.batch)

        # Project to latent space
        x = self.mlp(x)

        return x
