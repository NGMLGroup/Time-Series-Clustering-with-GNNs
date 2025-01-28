import torch
import numpy as np
import pandas.tseries.frequencies as pd_freq
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import rbf_kernel
from tsl.ops.similarities import top_k
from torch_geometric.utils import dense_to_sparse
from pygsp import graphs
from tqdm import tqdm


def construct_adjacency(dataset, adj_type='identity', mask=None,
                        train_slice=None, **kwargs):
    """
    Constructs an adjacency matrix for a given dataset based on the specified adjacency type.
    Parameters:
    -----------
    dataset : object (e.g. DateTimeDataset from tsl)
        The dataset object containing the time series. Must have the
        following attributes:
        - n_nodes: Number of nodes in the graph.
        - dataframe: Method that returns the dataset as a pandas DataFrame.
                     Must have datetime index for select adj. types.
        - mask: Mask to apply to the dataset (can be None).
        - freq: Frequency of the time series.

    adj_type : str
        The type of adjacency matrix to construct. Options are:
        - 'identity': Identity matrix.
        - 'euclidean': Euclidean dissimilarity + knn.
        - 'pearson': Pearson correlation coefficient + knn.
        - 'correntropy': Correntropy similarity measure + knn.
        - 'full': Fully connected graph.
        - 'random': Random graph using Erdos-Renyi model.
    mask : array-like, optional
        Mask to apply to the dataset.
    train_slice : slice, optional
        Slice of the dataset to use for training.
    **kwargs : dict
        Additional keyword arguments for specific adjacency types:
        - 'gamma': (float) Parameter for 'correntropy' adjacency type.
        - 'knn': (int) Number of nearest neighbors for 'pearson' and
                'correntropy' adjacency types.
    Returns:
    --------
    adj : tuple of torch.Tensor
        The constructed adjacency matrix in sparse format
        (edge_index, edge_weight).
    """

    n_nodes = dataset.n_nodes
    train_df = dataset.dataframe()
    if mask is None:
        mask = dataset.mask
    train_df = train_df * mask[..., -1]
    if train_slice is not None:
        train_df = dataset.dataframe().iloc[train_slice]
    x = train_df.values

    if adj_type == 'identity':
        adj = np.eye(n_nodes, dtype=np.float32)
        adj = dense_to_sparse(torch.from_numpy(adj))

    elif adj_type == 'euclidean':
        adj = kneighbors_graph(x.T, n_neighbors=3, mode='connectivity',
                               include_self=False)
        # Make the adjacency matrix symmetric (undirected)
        adj = adj.maximum(adj.T)
        adj = adj.tocoo()
        # Force binary weights
        adj = (torch.tensor([adj.row, adj.col], dtype=torch.long),
               torch.ones(adj.nnz, dtype=torch.float32))

    elif adj_type == 'pearson' or adj_type == 'correntropy':

        if adj_type == 'pearson':
            tot = train_df.mean(1).to_frame()
            bias = tot.groupby([tot.index.weekday,
                                tot.index.hour,
                                tot.index.minute]).transform(np.nanmean).values
            scale = train_df.values.std(0, keepdims=True)
            train_df = train_df - bias * scale
            adj = np.corrcoef(train_df.values, rowvar=False)

        elif adj_type == 'correntropy':
            gamma = kwargs.get('gamma', 0.05)

            x = (x - x.mean()) / x.std()
            # one week
            period = pd_freq.to_offset('7D').nanos // dataset.freq.nanos
            chunks = range(period, len(x), period)
            adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
            for i in tqdm(chunks, desc="Computing correntropy for every week"):
                xi = x[i - period:i].T
                adj += rbf_kernel(xi, gamma=gamma)
            adj /= len(range(period, len(x), period))


        knn = kwargs.get('knn', 3)

        if knn is not None:
            adj = top_k(adj,
                        knn,
                        include_self=False, # Do not include self-loops
                        keep_values=False) # Binary adjacency matrix
        adj = np.maximum.reduce([adj, adj.T]) # Symmetrize

        adj = dense_to_sparse(torch.from_numpy(adj))

    elif adj_type == 'full':
        adj = np.ones((n_nodes, n_nodes), dtype=np.float32)
        adj = dense_to_sparse(torch.from_numpy(adj))

    elif adj_type == 'random':
        graph = graphs.ErdosRenyi(N=n_nodes, p=0.005, seed=0,
                    directed=False, self_loops=False, connected=False,
                    max_iter=20)
        adj = graph.W.tocoo()
        adj = torch.tensor([adj.row, adj.col], dtype=torch.long)
        adj = (adj, torch.ones(adj.size(1), dtype=torch.float32))

    else:
        raise ValueError(f"Adjacency type {adj_type} not recognized")

    return adj



if __name__ == '__main__':
    # Example usage with Elergone dataset
    from tsl.datasets import Elergone

    dataset = Elergone(root='./../data/elergone', freq='15min')
    adj = construct_adjacency(dataset, adj_type='pearson')
    print(adj)

    # NOTE: connected is set to False for 'random' since Elergone has fewer
    # number of nodes than CER, so it takes many more trials to generate a
    # connected graph when p=0.005.