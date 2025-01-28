import math
import pprint
import os.path
import torch

import numpy as np

from einops import rearrange
from torch import nn
from torch_geometric.utils import remove_self_loops, coalesce
from pygsp import graphs
from tsl import logger
from tsl.datasets import GaussianNoiseSyntheticDataset
from tsl.nn.layers.graph_convs.gpvar import GraphPolyVAR
from tsl.ops.connectivity import parse_connectivity


class _SineGPVAR(nn.Module):
    def __init__(self,
                 temporal_order,
                 spatial_order,
                 num_nodes,
                 cluster_index=None):
        super(_SineGPVAR, self).__init__()
        self.global_model = GraphPolyVAR(temporal_order=temporal_order,
                                         spatial_order=spatial_order,
                                         norm='none')

        self.temporal_order = temporal_order
        self.spatial_order = spatial_order
        self.num_nodes = num_nodes
        self.cluster_index = cluster_index

        if self.cluster_index is None:
            self.amplitude = nn.Parameter(torch.Tensor(num_nodes, 1))
            self.phase = nn.Parameter(torch.Tensor(num_nodes, 1))
            self.period = nn.Parameter(torch.Tensor(num_nodes, 1))

        else:
            num_clusters = torch.unique(self.cluster_index).numel()
            self.amplitude = nn.Parameter(torch.Tensor(num_clusters, 1))
            self.phase = nn.Parameter(torch.Tensor(num_clusters, 1))
            self.period = nn.Parameter(torch.Tensor(num_clusters, 1))

        self.time_idx = 0

        self.reset_parameters()

    def reset_parameters(self):
        self.global_model.reset_parameters()
        amp_max = 1.
        phase_max = math.pi
        period_max = 12
        with torch.no_grad():
            self.amplitude.data.uniform_(0, amp_max)
            self.phase.data.uniform_(-phase_max, phase_max)
            self.period.data.uniform_(1, period_max)

    @classmethod
    def from_params(cls, global_params, local_params=None, num_nodes=None,
                    amp_max=1., phase_max=math.pi, period_max=12,
                    cluster_index=None, seed=None):

        temporal_order = global_params.shape[1]
        spatial_order = global_params.shape[0] - 1  #
        num_nodes = num_nodes or local_params.shape[0]

        model = cls(temporal_order=temporal_order,
                    spatial_order=spatial_order,
                    num_nodes=num_nodes,
                    cluster_index=cluster_index)
        model.global_model.weight.data.copy_(global_params)

        if local_params is None:

            rng = torch.Generator()
            if seed is not None:
                rng.manual_seed(seed)

            if cluster_index is None:
                num_instances = num_nodes
            else:
                num_instances = torch.unique(cluster_index).numel()

            amplitude = amp_max * torch.rand(num_instances, 1, generator=rng)
            phase = 2*phase_max * torch.rand(num_instances, 1, generator=rng) - phase_max
            period = period_max * torch.rand(num_instances, 1, generator=rng) + 1

        else:
            amplitude = local_params['amplitude']
            phase = local_params['phase']
            period = local_params['period']

        model.amplitude.data.copy_(amplitude)
        model.phase.data.copy_(phase)
        model.period.data.copy_(period)

        return model

    def forward(self, x, edge_index, edge_weight=None):
        # x : [batch, steps, nodes, channels]
        x_l = torch.tanh(self.global_model(x, edge_index, edge_weight))

        amplitude = self.amplitude
        phase = self.phase
        period = self.period
        if self.cluster_index is not None:
            amplitude = amplitude[self.cluster_index]
            phase = phase[self.cluster_index]
            period = period[self.cluster_index]

        t = self.time_idx
        sine_wave = amplitude * torch.sin(2 * math.pi * t / period + phase)
        sine_wave = rearrange(sine_wave, 'n 1 -> 1 1 n 1')
        self.time_idx += 1

        return x_l + sine_wave


class _ARProcess(nn.Module):
    def __init__(self, temporal_order, num_nodes, cluster_index=None):
        super(_ARProcess, self).__init__()
        self.temporal_order = temporal_order
        self.num_nodes = num_nodes
        self.cluster_index = cluster_index

        if self.cluster_index is None:
            self.weight = nn.Parameter(torch.Tensor(num_nodes, temporal_order))
        else:
            num_clusters = torch.unique(self.cluster_index).numel()
            self.weight = nn.Parameter(torch.Tensor(num_clusters, temporal_order))

        self.reset_parameters()

    def reset_parameters(self):
        a = math.sqrt(self.weight.size(1))
        with torch.no_grad():
            self.weight.data.uniform_(-a, a)

    @classmethod
    def from_params(cls, num_nodes, temporal_order, ar_weights = None,
                    p_max=1., cluster_index=None,
                    seed=None):

        model = cls(temporal_order=temporal_order, num_nodes=num_nodes,
                    cluster_index=cluster_index)

        if ar_weights is None:
            rng = torch.Generator()
            if seed is not None:
                rng.manual_seed(seed)

            if cluster_index is None:
                num_instances = num_nodes
            else:
                num_instances = torch.unique(cluster_index).numel()

            if p_max > 0:
                ar_weights = 2. * p_max * torch.rand(num_instances, temporal_order,
                                                    generator=rng) - p_max
            else:
                ar_weights = torch.ones(num_instances,
                                        temporal_order) / temporal_order
        else:
            assert temporal_order == ar_weights.shape[1]

        model.weight.data.copy_(ar_weights)
        return model

    def forward(self, x):
        weight = self.weight
        if self.cluster_index is not None:
            weight = weight[self.cluster_index]
        x_l = torch.einsum('bpnf, np -> bnf', x,
                           weight)
        x_l = rearrange(x_l, 'b n f -> b 1 n f')
        return x_l


def _mixed_barabasialbert_graph(num_nodes, num_communities, graph_params=None, community_prop=None, seed=None):
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)

    if num_nodes is None:
        num_nodes = int(100*num_communities)
    if community_prop is None:
        cluster_index = torch.arange(num_communities).repeat_interleave(int(num_nodes/num_communities))
        cluster_sizes = torch.tensor([int(num_nodes/num_communities)]*num_communities)
    else:
        cluster_sizes = (community_prop * num_nodes).int()
        cluster_index = torch.cat([i * torch.ones(s) for i, s in enumerate(cluster_sizes)]).int()

    if graph_params is None:
        m0_vals = []
        m_vals = []
        for i in range(num_communities):
            m0_vals.append(torch.randint(1, 10 if cluster_sizes[i] >= 10 else cluster_sizes[i],
                                            (1,), generator=rng).item())
            if m0_vals[-1] > 1:
                m_vals.append(torch.randint(1, m0_vals[-1],
                                            (1,), generator=rng).item())
            else:
                m_vals.append(1)
    else:
        m0_vals = graph_params['m0']
        m_vals = graph_params['m']

    edge_index = torch.tensor([], dtype=torch.long)
    for i in range(num_communities):
        graph = graphs.BarabasiAlbert(N=cluster_sizes[i], m0=m0_vals[i],
                                      m=m_vals[i], seed=seed)
        edge_index_i = graph.W.tocoo()
        edge_index_i = torch.tensor([edge_index_i.row, edge_index_i.col],
                                    dtype=torch.long)
        edge_index_i += cluster_sizes[:i].sum()
        edge_index = torch.cat([edge_index, edge_index_i], dim=1)

    n_noise_edges = (num_nodes // 10)
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)
    noise_edges = torch.randint(0, num_nodes, (2, n_noise_edges), generator=rng)
    # Concatenate in both directions
    edge_index = torch.cat([edge_index, noise_edges], dim=1)
    edge_index = torch.cat([edge_index, noise_edges.flip(0)], dim=1)
    # Remove potential duplicates and self-loops
    edge_index = remove_self_loops(edge_index)[0]
    edge_index = coalesce(edge_index, None, num_nodes)

    return cluster_index, edge_index


def _erdosrenyi_graph(num_nodes, num_communities, graph_params, community_prop=None, seed=None):

    if graph_params['p'] is None:
        p = 0.05
    else:
        p = graph_params['p']
    if graph_params['connected'] is None:
        connected = True
    else:
        connected = graph_params['connected']
    if graph_params['max_iter'] is None:
        max_iter = 40
    else:
        max_iter = graph_params['max_iter']

    if num_nodes is None:
        num_nodes = int(100*num_communities)
    if community_prop is None:
        cluster_index = torch.arange(num_communities).repeat_interleave(int(num_nodes/num_communities))
    else:
        cluster_sizes = (community_prop * num_nodes).int()
        cluster_index = torch.cat([i * torch.ones(s) for i, s in enumerate(cluster_sizes)]).int()

    graph = graphs.ErdosRenyi(N=num_nodes, p=p, seed=seed,
                                directed=False, self_loops=False, connected=connected,
                                max_iter=max_iter)
    edge_index = graph.W.tocoo()
    edge_index = torch.tensor([edge_index.row, edge_index.col],
                                dtype=torch.long)

    return cluster_index, edge_index


class SyntheticSpatioTemporalDataset(GaussianNoiseSyntheticDataset):
    def __init__(self,
                 num_nodes=None,
                 num_communities=None,
                 num_steps=None,
                 global_params=None,
                 series_type='sine_gpvar',
                 graph_type='mixed_ba',
                 local_params=None,
                 graph_params=None,
                 community_prop=None,
                 local_limits=None, # Different limits of local params
                 sigma_noise=.2,
                 share_community_weights: bool = False,
                 save_to: str = None,
                 load_from: str = None,
                 seed: int = None,
                 name=None):
        if name is None:
            self.name = f"SynthSpatioTemporal"
        else:
            self.name = name
        self.load_from = load_from

        if load_from is None:
            if seed is not None:
                self.seed = seed

            if graph_type == 'mixed_ba':
                cluster_index, edge_index = _mixed_barabasialbert_graph(num_nodes,
                                                                                num_communities,
                                                                                graph_params,
                                                                                community_prop,
                                                                                self.seed)
            elif graph_type == 'erdosrenyi':
                cluster_index, edge_index = _erdosrenyi_graph(num_nodes, num_communities, graph_params, community_prop, seed=self.seed)
            else:
                raise ValueError(f'Unknown graph type: {graph_type}')

            self.cluster_index = cluster_index

        else:
            params_dict = np.load(load_from + '/dataset_params.npy', allow_pickle='TRUE').item()

            num_nodes = params_dict['num_nodes']
            num_communities = params_dict['num_communities']
            num_steps = params_dict['num_steps']
            global_params = params_dict['global_params']
            series_type = params_dict['series_type']
            graph_type = params_dict['graph_type']
            local_params = params_dict['local_params']
            graph_params = params_dict['graph_params']
            community_prop = params_dict['community_prop']
            local_limits = params_dict['local_limits']
            sigma_noise = params_dict['sigma_noise']
            share_community_weights = params_dict['share_community_weights']
            seed = params_dict['seed']

            self.seed = seed
            self.cluster_index = torch.from_numpy(self.load_cluster_index())
            edge_index = torch.from_numpy(self.load_edge_index())


        if series_type == 'sine_gpvar':
            if local_limits is not None:
                amp_max = local_limits['amp_max']
                phase_max = local_limits['phase_max']
                period_max = local_limits['period_max']
            else:
                amp_max = 2.
                phase_max = math.pi
                period_max = 12

            filter = _SineGPVAR.from_params(
                global_params=torch.tensor(global_params,
                                        dtype=torch.float32),
                local_params=local_params,
                num_nodes=num_nodes,
                amp_max=amp_max,
                phase_max=phase_max,
                period_max=period_max,
                cluster_index=self.cluster_index if share_community_weights else None,
                seed=self.seed)

        elif series_type == 'arprocess':
            if local_limits is not None:
                p_max = local_limits['p_max']
            else:
                p_max = 1.
            filter = _ARProcess.from_params(
                num_nodes=num_nodes,
                temporal_order=global_params,
                ar_weights=local_params,
                p_max=p_max,
                cluster_index=self.cluster_index if share_community_weights else None,
                seed=self.seed)
        else:
            raise ValueError(f'Unknown series type: {series_type}')

        temporal_order = filter.temporal_order

        super(SyntheticSpatioTemporalDataset, self).__init__(num_features=1,
                                            num_nodes=num_nodes,
                                            num_steps=num_steps,
                                            connectivity=edge_index,
                                            min_window=temporal_order,
                                            model=filter,
                                            sigma_noise=sigma_noise,
                                            seed=seed,
                                            name=name)


        if save_to is not None:
            params_dict = {'num_nodes': num_nodes,
                            'num_communities': num_communities,
                            'num_steps': num_steps,
                            'global_params': global_params,
                            'series_type': series_type,
                            'graph_type': graph_type,
                            'local_params': local_params,
                            'graph_params': graph_params,
                            'community_prop': community_prop,
                            'local_limits': local_limits,
                            'sigma_noise': sigma_noise,
                            'share_community_weights': share_community_weights,
                            'seed': seed}

            self.save(foldername=save_to, params_dict=params_dict)


    def save(self, foldername: str, params_dict: dict):
        if not hasattr(self, 'target'):
            target, optimal_pred, mask = self.load_raw()
        else:
            target, optimal_pred, mask = self.target, self.optimal_pred, self.mask
        if not os.path.isabs(foldername):
            this_dir = os.path.dirname(os.path.realpath(__file__))
            foldername = os.path.join(this_dir, foldername)
        if not os.path.exists(foldername):
            os.makedirs(foldername)

        logger.info(f'Saving synthetic dataset to: {foldername}')

        # Save target, optimal_pred, mask
        np.savez_compressed(foldername + '/series.npz', target=target,
                            optimal_pred=optimal_pred, mask=mask)

        # Save dataset paramameters (as both .npy and .txt)
        np.save(foldername + '/dataset_params.npy', params_dict)
        with open(foldername + '/dataset_params.txt', 'w') as f:
            pprint.pprint(params_dict, stream=f)

        # Save cluster labels
        np.save(foldername + '/cluster_index.npy', self.cluster_index.numpy())

        # Save edge_index
        np.save(foldername + '/edge_index.npy', self.connectivity[0].numpy())


    def load(self):
        if self.load_from is None:
            return self.load_raw()
        else:
            filename = self.load_from + '/series.npz'
            if not os.path.isabs(filename):
                this_dir = os.path.dirname(os.path.realpath(__file__))
                filename = os.path.join(this_dir, filename)
            logger.warning(f'Loading synthetic dataset from: {filename}')
            content = np.load(filename)
            target = content['target']
            optimal_pred = content['optimal_pred']
            mask = content['mask']
            return target, optimal_pred, mask

    def load_edge_index(self):
        filename = self.load_from + '/edge_index.npy'
        if not os.path.isabs(filename):
            this_dir = os.path.dirname(os.path.realpath(__file__))
            filename = os.path.join(this_dir, filename)
        logger.warning(f'Loading edge index from: {filename}')
        return np.load(filename)

    def load_cluster_index(self):
        filename = self.load_from + '/cluster_index.npy'
        if not os.path.isabs(filename):
            this_dir = os.path.dirname(os.path.realpath(__file__))
            filename = os.path.join(this_dir, filename)
        logger.warning(f'Loading cluster index from: {filename}')
        return np.load(filename)

    def get_connectivity(self, include_weights: bool = True,
                         include_self: bool = True,
                         layout: str = 'edge_index', **kwargs):
        edge_index, edge_weight = self.connectivity
        if not include_weights:
            edge_weight = None
        if not include_self:
            edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        connectivity = (edge_index, edge_weight)
        if layout in ['coo', 'csr', 'sparse_matrix']:
            return parse_connectivity(connectivity=connectivity,
                                      target_layout='sparse',
                                      num_nodes=self.n_nodes)
        elif layout == 'dense':
            return parse_connectivity(connectivity=connectivity,
                                      target_layout='dense',
                                      num_nodes=self.n_nodes)
        else:
            return connectivity


if __name__ == '__main__':
    # Syntax for generating the dataset
    dataset = SyntheticSpatioTemporalDataset(
        series_type='sine_gpvar',
        graph_type='mixed_ba',
        num_communities=5,
        num_nodes=1500,
        num_steps=2000,
        community_prop=None,
        graph_params={'m0': [3, 2, 5, 2, 4], 'm': [2, 1, 4, 1, 2]},
        global_params=[[0.6, 0.],
                    [0., 0.2]],
        local_params={'amplitude': torch.tensor([[0.4], [0.5], [0.6], [0.4], [0.5]]),
                    'phase': torch.tensor([[0.], [0.15], [-0.15], [-0.2], [0.2]]),
                    'period': torch.tensor([[12.], [14.], [10.], [16.], [12.]])},
        sigma_noise=0.6,
        seed=42,
        share_community_weights=True,
        save_to='../data/synthetic/balanced',
    )

    # Syntax for loading the dataset
    # dataset = SyntheticSpatioTemporalDataset(
    #     load_from='../data/synthetic/balanced',
    # )