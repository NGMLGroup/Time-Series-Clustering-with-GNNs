import math
import torch

from typing import Optional, Tuple
from torch import Tensor


# Pooling functions with modified node feature pooling (normalized by the number
# of nodes in each cluster) and temperature softmax. The coarsened adjacency
# is not returned.


# Based on PyTorch Geometric implementation
def dense_mincut_pool(
    x: Tensor,
    adj: Tensor,
    s: Tensor,
    mask: Optional[Tensor] = None,
    temp: float = 1.0,
) -> Tuple[Tensor, Tensor, Tensor]:

    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    (batch_size, num_nodes, _), k = x.size(), s.size(-1)

    s = torch.softmax(s / temp if temp != 1.0 else s, dim=-1)

    if mask is not None:
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask

    out = torch.matmul((s/s.sum(dim=-2, keepdim=True)).transpose(-2, -1), x)
    coarse_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

    # MinCut regularization.
    mincut_num = _rank3_trace(coarse_adj)
    d_flat = torch.einsum('ijk->ij', adj)
    d = _rank3_diag(d_flat)
    mincut_den = _rank3_trace(
        torch.matmul(torch.matmul(s.transpose(1, 2), d), s))
    mincut_loss = -(mincut_num / mincut_den)
    mincut_loss = torch.mean(mincut_loss)

    # Orthogonality regularization.
    ss = torch.matmul(s.transpose(1, 2), s)
    i_s = torch.eye(k).type_as(ss)
    ortho_loss = torch.norm(
        ss / torch.norm(ss, dim=(-1, -2), keepdim=True) -
        i_s / torch.norm(i_s), dim=(-1, -2))
    ortho_loss = torch.mean(ortho_loss)

    return out, mincut_loss, ortho_loss


def _rank3_trace(x: Tensor) -> Tensor:
    return torch.einsum('ijj->i', x)


def _rank3_diag(x: Tensor) -> Tensor:
    eye = torch.eye(x.size(1)).type_as(x)
    out = eye * x.unsqueeze(2).expand(x.size(0), x.size(1), x.size(1))

    return out


# Based on PyTorch Geometric implementation
def dense_diff_pool(
    x: Tensor,
    adj: Tensor,
    s: Tensor,
    mask: Optional[Tensor] = None,
    normalize: bool = True,
    temp: float = 1.0,
) -> Tuple[Tensor, Tensor, Tensor]:

    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    batch_size, num_nodes, _ = x.size()

    s = torch.softmax(s / temp if temp != 1.0 else s, dim=-1)

    if mask is not None:
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask

    out = torch.matmul((s/s.sum(dim=-2, keepdim=True)).transpose(-2, -1), x)

    link_loss = adj - torch.matmul(s, s.transpose(1, 2))
    link_loss = torch.norm(link_loss, p=2)
    if normalize is True:
        link_loss = link_loss / adj.numel()

    ent_loss = (-s * torch.log(s + 1e-15)).sum(dim=-1).mean()

    return out, link_loss, ent_loss


# Based on PyTorch Geometric implementation
def dense_dmon_pool(
    x: Tensor,
    adj: Tensor,
    s: Tensor,
    mask: Optional[Tensor] = None,
    temp: float = 1.0,
    ) -> Tuple[Tensor, Tensor, Tensor]:

        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        s = s.unsqueeze(0) if s.dim() == 2 else s

        s = torch.softmax(s / temp if temp != 1.0 else s, dim=-1)

        (batch_size, num_nodes, _), C = x.size(), s.size(-1)

        if mask is None:
            mask = torch.ones(batch_size, num_nodes, dtype=torch.bool,
                              device=x.device)

        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask

        out = torch.matmul((s/s.sum(dim=-2, keepdim=True)).transpose(-2, -1), x)
        coarse_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

        # Spectral loss:
        degrees = torch.einsum('ijk->ij', adj)
        degrees = degrees.unsqueeze(-1) * mask
        degrees_t = degrees.transpose(1, 2)

        m = torch.einsum('ijk->i', degrees) / 2
        m_expand = m.view(-1, 1, 1).expand(-1, C, C)

        ca = torch.matmul(s.transpose(1, 2), degrees)
        cb = torch.matmul(degrees_t, s)

        normalizer = torch.matmul(ca, cb) / 2 / m_expand
        decompose = coarse_adj - normalizer
        spectral_loss = -_rank3_trace(decompose) / 2 / m
        spectral_loss = spectral_loss.mean()

        # Cluster loss:
        i_s = torch.eye(C).type_as(s)
        cluster_size = torch.einsum('ijk->ik', s)
        cluster_loss = torch.norm(input=cluster_size, dim=1)
        cluster_loss = cluster_loss / mask.sum(dim=1) * torch.norm(i_s) - 1
        cluster_loss = cluster_loss.mean()

        return out, spectral_loss, cluster_loss

# Based on implementation in:
# https://github.com/FilippoMB/Total-variation-graph-neural-networks
def dense_asymcheegercut_pool(
        x: Tensor, adj: Tensor, s: Tensor, mask: Optional[Tensor] = None,
        temp: float = 1.0
) -> Tuple[Tensor, Tensor, Tensor]:

    def _totvar_loss(adj, s):
        l1_norm = torch.sum(torch.abs(s[..., None, :] - s[:, None, ...]),
                            dim=-1)

        loss = torch.sum(adj * l1_norm, dim=(-1, -2))

        # Normalize loss
        n_edges = torch.count_nonzero(adj, dim=(-1, -2))
        loss *= 1 / (2 * n_edges)

        return loss

    def _balance_loss(s):
        n_nodes, n_clust = s.size()[-2], s.size()[-1]

        # k-quantile
        idx = int(math.floor(n_nodes / n_clust))
        quant = torch.sort(s, dim=-2, descending=True)[0][:, idx, :]

        # Asymmetric l1-norm
        loss = s - torch.unsqueeze(quant, dim=1)
        loss = (loss >= 0) * (n_clust - 1) * loss + (loss < 0) * loss * -1
        loss = torch.sum(loss, dim=(-1, -2))
        loss = 1 / (n_nodes * (n_clust - 1)) * (n_nodes * (n_clust - 1) - loss)

        return loss


    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    s = torch.softmax(s / temp if temp != 1.0 else s, dim=-1)

    batch_size, n_nodes = x.size(0), x.size(-2)

    if mask is not None:
        mask = mask.view(batch_size, n_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask

    # Pooled features and adjacency
    out = torch.matmul((s/s.sum(dim=-2, keepdim=True)).transpose(-2, -1), x)

    # Total variation loss
    tv_loss = torch.mean(_totvar_loss(adj, s))

    # Balance loss
    bal_loss = torch.mean(_balance_loss(s))

    return out, tv_loss, bal_loss
