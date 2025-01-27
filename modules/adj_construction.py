import torch
import tqdm
import numpy as np
import pandas.tseries.frequencies as pd_freq
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import rbf_kernel
from tsl.ops.similarities import top_k
from torch_geometric.utils import dense_to_sparse
from pygsp import graphs



def construct_adjacency(dataset, adj_type='identity', mask=None, 
                        train_slice=None, **kwargs):
    n_nodes = dataset.n_nodes
    if adj_type == 'identity':
        adj = np.eye(n_nodes, dtype=np.float32)
        adj = dense_to_sparse(torch.from_numpy(adj))

    elif adj_type == 'euclidean':
        adj = kneighbors_graph(x.T[0], n_neighbors=3, mode='connectivity', 
                               include_self=False)
        # Make the adjacency matrix symmetric (undirected)
        adj = adj.maximum(adj.T)  
        adj = adj.tocoo()
        # Force binary weights
        adj = (torch.tensor([adj.row, adj.col], dtype=torch.long), 
               torch.ones(adj.nnz, dtype=torch.float32))  

    elif adj_type == 'pearson' or adj_type == 'correntropy':
        train_df = dataset.dataframe()
        if mask is None:
            mask = dataset.mask
        train_df = train_df * mask[..., -1]
        if train_slice is not None:
            train_df = dataset.dataframe().iloc[train_slice]

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
        
        adj = (torch.from_numpy(adj[0]), 
               torch.from_numpy(adj[1]).type(torch.float32))

    elif adj_type == 'full':
        adj = np.ones((n_nodes, n_nodes), dtype=np.float32)
        adj = dense_to_sparse(torch.from_numpy(adj))

    elif adj_type == 'random':
        graph = graphs.ErdosRenyi(N=n_nodes, p=0.005, seed=0,
                    directed=False, self_loops=False, connected=True,
                    max_iter=20)
        adj = graph.W.tocoo()
        adj = torch.tensor([adj.row, adj.col], dtype=torch.long)
        adj = (adj, torch.ones(adj.size(1), dtype=torch.float32))


    else:
        raise ValueError(f"Adjacency type {adj_type} not recognized")
    
    return adj




if __name__ == '__main__':
    # TODO: Check if this works
    import matplotlib.pyplot as plt
    from tsl.datasets import SolarEnergy
    from tsl.utils import plot_adjacency

    dataset = SolarEnergy()
    adj = construct_adjacency(dataset, adj_type='pearson')
    plot_adjacency(adj, dataset.n_nodes)
    plt.show()