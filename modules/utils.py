from typing import Optional, Tuple, List, Union
from collections import Counter
import os
import math
from einops import rearrange
import random
import itertools
import torch
from torch import Tensor
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import scatter
from torch_geometric.typing import Adj
from torch_sparse import SparseTensor
from tsl.data.preprocessing import StandardScaler
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.stats import gaussian_kde
from scipy.cluster.hierarchy import dendrogram, fcluster
from sklearn.metrics import normalized_mutual_info_score as nmi_score
from torch_geometric.profile import get_gpu_memory_from_nvidia_smi


def softmax_with_temperature(x, dim=-1, t=1.0):
    return torch.softmax(x / t if t != 1.0 else x, dim=dim)


def straight_through_softmax(x: Tensor, dim: int = -1, t: float = 1.0):
    soft = torch.softmax(x / t if t != 1.0 else x, dim=dim)
    _, ind = soft.max(dim=dim, keepdim=True)
    hard = torch.zeros_like(x).scatter_(dim, ind, 1.0)
    return hard - soft.detach() + soft
