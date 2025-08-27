import torch

from einops import rearrange
from torch_geometric.utils import to_dense_adj
from torch_geometric.typing import OptTensor
from tsl.nn.blocks import MLPDecoder
from tsl.nn.blocks.encoders import ConditionalBlock, RNN
from .layers import GNNEncoder, PoolingLayerWithStaticAssignments, DilatedTCN
from .utils import softmax_with_temperature, straight_through_softmax

class TTSModel(torch.nn.Module):
    """Time Then Space STGNN model.
    Args:
        input_size : int
            Number of input features/channels.
        exog_size : int
            Number of exogenous features (covariates).
        hidden_size : int
            Number of hidden units.
        temporal_layers : int
            Number of layers in the temporal encoder.
        kernel_size : int
            Size of the kernel in the temporal encoder.
        dilation : int
            Dilation factor for the temporal encoder.
        exponential_dilation : bool
            Whether to use exponential dilation.
        skip_connection : bool
            Whether to use skip connections in the temporal encoder.
        gnn_layers : int
            Number of message passing layers in the graph neural network
            encoder.
        n_nodes : int
            Number of nodes in the graph.
        n_clusters : int
            Number of clusters in the pooling layer.
        topo_w : float
            Weight of the topological loss.
        qual_w : float
            Weight of the quality loss.
        horizon : int
            Prediction horizon.
        temporal_enc_type : str
            Type of temporal encoder to use. Options: 'tcn', 'gru'.
        temporal_aggr_type : str
            Type of temporal aggregation to use. Options: 'attention', 'mean',
            'last'.
        mp_method : str
            Message passing method. Options: 'gcs', 'gat'.
        pool_method : str
            Pooling method (pooling loss type). Options: 'mincut',
            'asymcheegercut', 'diffpool', 'dmon'.
        lift_softmax : str
            Softmax type applied during lifting (forward pass).
            Options: 'temperature', 'straight_through'.
        softmax_temp : float
            Initial softmax temperature.
    """
    def __init__(self,
                 input_size,
                 exog_size,
                 hidden_size,
                 temporal_layers,
                 kernel_size,
                 dilation,
                 exponential_dilation,
                 skip_connection,
                 gnn_layers,
                 n_nodes,
                 n_clusters,
                 topo_w,
                 qual_w,
                 horizon,
                 temporal_enc_type = 'tcn',
                 temporal_aggr_type = 'attention',
                 mp_method = 'gcs',
                 pool_method = 'mincut',
                 lift_softmax = 'temperature',
                 softmax_temp = 1.
                 ):
        super(TTSModel, self).__init__()

        if exog_size:
            self.input_encoder = ConditionalBlock(
                                    input_size=input_size,
                                    exog_size=exog_size,
                                    output_size=hidden_size,
                                    activation='relu'
                                )
        else:
            self.input_encoder = torch.nn.Linear(input_size, hidden_size)

        if temporal_enc_type == 'tcn':
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

        elif temporal_enc_type == 'gru':
            self.temporal_encoder = RNN(
                                    input_size=hidden_size,
                                    hidden_size=hidden_size,
                                    cell='gru',
                                    n_layers=temporal_layers,
                                    dropout=0.,
                                    return_only_last_state=False
                                )

        else:
            print(f"Unknown temporal encoder type: {temporal_enc_type}")

        self.temporal_aggr_type = temporal_aggr_type
        if self.temporal_aggr_type == 'attention':
                self.lin_q = torch.nn.Linear(hidden_size, hidden_size)
                self.lin_k = torch.nn.Linear(hidden_size, hidden_size,
                                             bias=False)



        self.gnn_encoder = GNNEncoder(
                                input_size=hidden_size,
                                hidden_size=hidden_size,
                                n_layers=gnn_layers,
                                dropout=0.,
                                mp_method=mp_method
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

        if lift_softmax == 'temperature':
            self.lift_softmax = softmax_with_temperature
        elif lift_softmax == 'straight_through':
            self.lift_softmax = straight_through_softmax

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
        x = self.lift(x, s)

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


    def lift(self, x, s):
        # x_in: [batches out_steps n_latent channels]

        s = self.lift_softmax(s, dim=-1, t=self.softmax_temp)

        # x_out: [batches out_steps nodes channels]
        x = torch.matmul(torch.unsqueeze(s, dim=1), x)

        return x


    def get_latent_factors(self, x, edge_index, edge_weight = None, u = None):
        x, s, aux_loss = self.encode(x, edge_index, edge_weight, u)
        x = self.temporal_decoder(x)
        return x, s, aux_loss


    def temporal_aggr(self, x: torch.Tensor, dim: int = -3):

        if isinstance(self.temporal_encoder, DilatedTCN):
            # Removes the first entries because they include padded values in the
            # receptive field.
            # E.g. if the sequence length is 48 and the recept field is 27, we keep
            # only the last 22 entries.
            if x.size(dim) > self.receptive_field:
                index = torch.arange(self.receptive_field - 1, x.size(dim),
                                    device=x.device)
                x = x.index_select(dim, index)

        if self.temporal_aggr_type == 'mean':
            return x.mean(dim)
        elif self.temporal_aggr_type == 'last':
            return x[:, -1]
        elif self.temporal_aggr_type == 'attention':
            # Apply linear self-attention
            q = self.lin_q(torch.select(x, dim, -1)).unsqueeze(dim)
            k = self.lin_k(x)
            alpha = torch.einsum('bqnf,btnf->btnf', q, k).softmax(dim=dim)
            return (x * alpha).sum(dim)
        else:
            raise ValueError("Unknown temporal aggregation type: "
                             f"{self.temporal_aggr_type}")
