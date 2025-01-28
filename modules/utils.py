import torch
from torch import Tensor

def softmax_with_temperature(x, dim=-1, t=1.0):
    return torch.softmax(x / t if t != 1.0 else x, dim=dim)

def straight_through_softmax(x: Tensor, dim: int = -1, t: float = 1.0):
    soft = torch.softmax(x / t if t != 1.0 else x, dim=dim)
    _, ind = soft.max(dim=dim, keepdim=True)
    hard = torch.zeros_like(x).scatter_(dim, ind, 1.0)
    return hard - soft.detach() + soft
