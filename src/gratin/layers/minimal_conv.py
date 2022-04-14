from .diverse import *
from torch_geometric.nn import MessagePassing
from torch_sparse import matmul


class MinimalJumpsConv(MessagePassing):
    def __init__(
        self,
        out_channels,
        x_dim,
        dropout=0.0,
        aggr="mean",
        moments=[1],
        f_inner_width=[128, 64],
        **kwargs
    ):
        super(MinimalJumpsConv, self).__init__(
            aggr=aggr, **kwargs
        )  # , flow="target_to_source"
        self.out_channels = out_channels
        self.p = dropout
        self.moments = moments
        M = len(moments) + 1
        self.bn_x = nn.BatchNorm1d(x_dim)

        MLP_size = [M * x_dim] + f_inner_width + [out_channels]
        if M > 1:
            self.f = nn.Sequential(
                nn.BatchNorm1d(M * x_dim), MLP(MLP_size, dropout=dropout)
            )  # last_activation=nn.Tanh()))
        else:
            self.f = MLP(MLP_size, dropout=dropout)

        nparams = 0
        for n, p in self.named_parameters():
            np_ = np.product(np.array([s for s in p.shape]))
            nparams += np_
        # print("f size = ", MLP_size)
        # print(
        #    "Convolution has %d parameters. Input dim is %d, output is %d"
        #    % (nparams, x_dim, out_channels)
        # )

    def forward(self, x, edge_index):  # removed edge_attr

        x = self.bn_x(x)

        neighbors_message = self.propagate(x=x, edge_index=edge_index)

        neighbors_message = torch.cat(
            [x] + [torch.pow(neighbors_message, m) for m in self.moments], dim=1
        )
        result = self.f(neighbors_message)

        return result

    def message_and_aggregate(self, adj_t, x):
        # print("message and aggregate")
        return matmul(adj_t, x, reduce=self.aggr)
