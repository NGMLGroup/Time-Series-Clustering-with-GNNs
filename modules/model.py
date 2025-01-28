import torch

from einops import rearrange
from torch_geometric.utils import to_dense_adj
from torch_geometric.typing import OptTensor

from tsl.nn.blocks import MLPDecoder
from tsl.nn.layers import NodeEmbedding
from tsl.nn.blocks.encoders import ConditionalBlock

from .layers import GNNEncoder, PoolingLayerWithStaticAssignments, DilatedTCN
from .utils import softmax_with_temperature, straight_through_softmax

class TTSModel(torch.nn.Module):
    def __init__(self,
                 input_size,
                 exog_size,
                 hidden_size,
                 temporal_layers,
                 kernel_size,
                 dilation,
                 exponential_dilation,
                 skip_connection,
                 gcn_layers,
                 n_nodes,
                 n_clusters,
                 topo_w,
                 qual_w,
                 horizon,
                 pool_method = 'mincut',
                 unpool_softmax = 'temperature', # Other option: 'straight_through'
                 softmax_temp = 1.
                 ):
        super(TTSModel, self).__init__()

        # TODO: Add docstring

        if exog_size:
            self.input_encoder = ConditionalBlock(
                                    input_size=input_size,
                                    exog_size=exog_size,
                                    output_size=hidden_size,
                                    activation='relu'
                                )
        else:
            self.input_encoder = torch.nn.Linear(input_size, hidden_size)

        self.temporal_encoder = DilatedTCN(
                                    input_size=hidden_size,
                                    hidden_size=hidden_size,
                                    n_layers=temporal_layers,
                                    dilation=dilation,
                                    exponential_dilation=exponential_dilation,
                                    kernel_size=kernel_size,
                                    ff_layers=0,
                                    skip_connection=skip_connection
                                )
        self.receptive_field = self.temporal_encoder.receptive_field
        self.lin_q = torch.nn.Linear(hidden_size, hidden_size)
        self.lin_k = torch.nn.Linear(hidden_size, hidden_size, bias=False)

        self.gnn_encoder = GNNEncoder(
                                input_size=hidden_size,
                                hidden_size=hidden_size,
                                n_layers=gcn_layers,
                                dropout=0.
                            )


        self.pooling_layer = PoolingLayerWithStaticAssignments(
                                k=n_clusters,
                                n_nodes=n_nodes,
                                topo_w=topo_w,
                                qual_w=qual_w,
                                method=pool_method
                            )

        self.temporal_decoder = MLPDecoder(
                                    input_size=hidden_size,
                                    hidden_size=hidden_size,
                                    output_size=input_size,
                                    activation="relu",
                                    horizon=horizon,
                                    dropout=0.
                                )

        if unpool_softmax == 'temperature':
            self.unpool_softmax = softmax_with_temperature
        elif unpool_softmax == 'straight_through':
            self.unpool_softmax = straight_through_softmax

        self.softmax_temp = softmax_temp



    def forward(self, x, edge_index, edge_weight = None, u: OptTensor = None):
        # x_in: [batches in_steps nodes channels]

        # x_out: [batches n_latent hidden_size]
        x, s, aux_loss = self.encode(x, edge_index, edge_weight, u)

        # x_out: [batches out_steps nodes channels]
        x = self.decode(x, s)

        return x, *aux_loss


    def encode(self, x, edge_index, edge_weight, u: OptTensor = None):
        # x_in: [batches in_steps nodes channels]

        # x_out: [batches in_steps nodes hidden_size]
        if u is not None:
            if u.dim() == 3:
                u = rearrange(u, 'b s c -> b s 1 c')
            x = self.input_encoder(x, u)
        else:
            x = self.input_encoder(x)

        # x_out: [batches nodes hidden_size]
        x = self.temporal_encoder(x)
        x = self.temporal_aggr(x)

        # x_out: [batches nodes hidden_size]
        x = self.gnn_encoder(x, edge_index, edge_weight)

        # x_out: [batches n_latent hidden_size]
        return self.pool(x, edge_index, edge_weight)


    def decode(self, x, s):
        # x_in: [batches n_latent hidden_size]

        # x_out: [batches out_steps n_latent channels]
        x = self.temporal_decoder(x)

        # x_out: [batches out_steps n_nodes channels]
        x = self.unpool(x, s)

        return x


    def pool(self, x, edge_index, edge_weight: OptTensor = None):
        # x_in: [batches nodes hidden_size]

        adj = to_dense_adj(edge_index=edge_index, edge_attr=edge_weight,
                           max_num_nodes=x.size(1))

        # x_out: [batches n_latent hidden_size]
        # s: [1 nodes n_clusters] (containing logits)
        x, s, aux_loss = self.pooling_layer(x, adj,
                                            return_assignment_mat=True,
                                            softmax_temp=self.softmax_temp)

        return x, s, aux_loss


    def unpool(self, x, s):
        # x_in: [batches out_steps n_latent channels]

        s = self.unpool_softmax(s, dim=-1, t=self.softmax_temp)

        # x_out: [batches out_steps nodes channels]
        x = torch.matmul(torch.unsqueeze(s, dim=1), x)

        return x


    def get_latent_factors(self, x, edge_index, edge_weight = None, u = None):
        x, s, aux_loss = self.encode(x, edge_index, edge_weight, u)
        x = self.temporal_decoder(x)
        return x, s, aux_loss


    def temporal_aggr(self, x: torch.Tensor, dim: int = -3):

        # Removes the first entries because they include padded values in the receptive field
        # E.g. if the sequence length is 48 and the recept field is 27, we keep only the last 22 entries
        if x.size(dim) > self.receptive_field:
            index = torch.arange(self.receptive_field - 1, x.size(dim),
                                 device=x.device)
            x = x.index_select(dim, index)

        # Apply linear self-attention
        q = self.lin_q(torch.select(x, dim, -1)).unsqueeze(dim)
        k = self.lin_k(x)
        alpha = torch.einsum('bqnf,btnf->btnf', q, k).softmax(dim=dim)
        return (x * alpha).sum(dim)
