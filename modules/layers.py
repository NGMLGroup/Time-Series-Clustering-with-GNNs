import math

import torch
from torch import nn, Tensor
from torch_geometric.nn.models.mlp import Linear
from torch_geometric.nn import MLP
from torch_geometric.nn.resolver import activation_resolver
from tsl.nn.layers import (GraphConv, TemporalConv, Concatenate, Activation,
                           Lambda, Dense, NodeEmbedding)
from tsl.nn import utils
from tsl import logger

from .pooling_functions import (dense_mincut_pool, dense_diff_pool,
                                dense_dmon_pool, dense_asymcheegercut_pool)

class GNNEncoder(torch.nn.Module):
    def __init__(self,
                input_size,
                hidden_size,
                n_layers=1,
                activation='relu',
                dropout=0.
                ):
        super(GNNEncoder, self).__init__()

        graph_convs = []
        for l in range(n_layers):
            graph_convs.append(
                GraphConv(input_size=input_size if l == 0 else hidden_size,
                        output_size=hidden_size,
                        root_weight=True, # skip connection
                        norm= 'sym',
                        cached=True,
                        bias=True
                        )
            )

        self.convs = torch.nn.ModuleList(graph_convs)
        self.activation = utils.get_functional_activation(activation)
        self.dropout = torch.nn.Dropout(dropout)


    def forward(self, h, edge_index, edge_weight=None):
        # h: [batches nodes features]
        for conv in self.convs:
            h = self.dropout(self.activation(conv(h, edge_index, edge_weight)))

        return h


class PoolingLayerWithStaticAssignments(torch.nn.Module):
    def __init__(self,
                 k,
                 n_nodes=None,          
                 topo_w=None,          
                 qual_w=None,          
                 method='mincut'     
                 ):
        """
        k: number of clusters
        mlp_units: list of units for the MLP
        activation: activation function
        bias: whether to use bias in the MLP
        static_assign: whether to use static or features-dependent cluster assignments
        n_nodes: number of nodes in the graph
        topo_w: weight of the first loss term
        qual_w: weight of second loss term
        method: pooling method

        For method = 'mincut', topo_w and qual_w are the weights of the cut and orthogonality loss terms, respectively.
        For method = 'asymcheegercut', topo_w is the gtv coefficient and qual_w is the balance coefficient.
        For method = 'diffpool', topo_w and qual_w are the weights of the link prediction and entropy loss terms, respectively.
        For method = 'dmonpool', topo_w and qual_w are the weights of the spectral and cluster loss terms, respectively.
        """
        super(PoolingLayerWithStaticAssignments, self).__init__()

        self.assignments = NodeEmbedding(n_nodes=n_nodes, emb_size=k)

        if method == 'mincut':
            self.pooling_op = dense_mincut_pool
        elif method == 'asymcheegercut':
            self.pooling_op = dense_asymcheegercut_pool
        elif method == 'diffpool':
            self.pooling_op = dense_diff_pool
        elif method == 'dmon':
            self.pooling_op = dense_dmon_pool
        else:
            raise NotImplementedError(f"Pooling method {method} not implemented")

        self.topo_w, self.qual_w = topo_w, qual_w


    def forward(self, x, adj, return_assignment_mat=False, softmax_temp=1.0):

        if x.dim() == 3:
            s = self.assignments().unsqueeze(0)

        # x: [batches nodes hidden_size]
        x, topo_loss, qual_loss = self.pooling_op(x, adj, s, temp=softmax_temp)
        aux_loss = [topo_loss*self.topo_w, qual_loss*self.qual_w]

        if return_assignment_mat:
            return x, s, aux_loss
        return x, aux_loss


class ReceptiveFieldLayer(nn.Module):

    def __init__(self, receptive_field: int = None):
        super().__init__()
        self.receptive_field = receptive_field
        self._hook = self.register_forward_pre_hook(
            self._check_receptive_field_hook, with_kwargs=True)

    def _check_receptive_field_hook(self, module, args, kwargs):
        x = args[0] if len(args) else kwargs['x']
        input_window = x.size(1)
        if input_window != self.receptive_field:
            logger.warning(
                f"Input is {input_window}-steps long, receptive field "
                f"is {self.receptive_field}")
        self._hook.remove()
        delattr(self, '_hook')


class DilatedTCN(ReceptiveFieldLayer):

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 n_layers: int,
                 dilation: int = 2,
                 exponential_dilation: bool = False,
                 kernel_size: int = 2,
                 ff_layers: int = 0,
                 skip_connection: bool = True,
                 activation: str = 'relu'):
        super().__init__(receptive_field=1)

        self.n_layers = n_layers
        if exponential_dilation:
            self.dilations = [dilation ** i for i in range(self.n_layers)]
        else:
            self.dilations = [dilation for _ in range(self.n_layers)]

        self.convs = nn.ModuleList()
        self.skip_connections = nn.ModuleList()

        for i, dilation in enumerate(self.dilations):
            module = [
                TemporalConv(
                    input_channels=input_size if i == 0 else hidden_size,
                    output_channels=hidden_size,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    channel_last=True
                ),
                Activation(activation),
            ]
            module += [
                Dense(hidden_size, hidden_size, activation=activation)
                for _ in range(ff_layers)
            ]
            self.convs.append(nn.Sequential(*module))
            self.receptive_field += dilation * (kernel_size - 1)
            if skip_connection:
                self.skip_connections.append(nn.Sequential(
                    Concatenate(dim=-1),
                    nn.Linear(in_features=input_size + hidden_size,
                              out_features=hidden_size),
                ))
            else:
                self.skip_connections.append(Lambda(lambda x: x[1]))

        self.readout = nn.Linear(in_features=n_layers, out_features=1)

    def forward(self, x: Tensor):
        # x: [batch, time, node, features]
        out = []
        x_in = x
        for conv, skip_conn in zip(self.convs, self.skip_connections):
            x_out = conv(x_in)
            out.append(x_out)
            x_in = skip_conn([x, x_out])

        out_stack = torch.stack(out, dim=1)
        out_read = self.readout(out_stack.movedim(1, -1)).squeeze(-1)
        return out_read
