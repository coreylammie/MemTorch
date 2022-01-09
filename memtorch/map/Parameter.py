import numpy as np
import torch
import torch.functional as F
import torch.nn as nn

import memtorch
from memtorch.utils import convert_range


def naive_map(weight, r_on, r_off, scheme, p_l=None):
    """Method to naively map network parameters to memristive device conductances, using two crossbars to represent both positive and negative weights.

    Parameters
    ----------
    weight : torch.Tensor
        Weight tensor to map.
    r_on : float
        Low resistance state.
    r_off : float
        High resistance state.
    scheme: memtorch.bh.crossbar.Scheme
        Weight representation scheme.
    p_l: float, optional
        If not None, the proportion of weights to retain.

    Returns
    -------
    torch.Tensor, torch.Tensor
        Positive and negative crossbar weights.
    """
    if p_l is not None:
        assert p_l >= 0 and p_l <= 1, "p_l must be None or between 0 and 1."
        weight_max = sorted(weight.abs().flatten().cpu().detach().numpy(), reverse=True)
        weight_max = weight_max[int(p_l * (weight.numel() - 1))]
        weight_min = weight_max / (r_off / r_on)
    else:
        weight_max = weight.abs().max()
        weight_min = 0

    if scheme == memtorch.bh.crossbar.Scheme.DoubleColumn:
        pos = weight.clone()
        neg = weight.clone() * -1
        pos[pos < 0] = 0
        neg[neg < 0] = 0
        pos = torch.clamp(pos, weight_min, weight_max)
        neg = torch.clamp(neg, weight_min, weight_max)
        pos = convert_range(pos, weight_min, weight_max, 1 / r_off, 1 / r_on)
        neg = convert_range(neg, weight_min, weight_max, 1 / r_off, 1 / r_on)
        return pos, neg
    elif scheme == memtorch.bh.crossbar.Scheme.SingleColumn:
        crossbar = weight.clone()
        crossbar = torch.clamp(crossbar, weight_min, weight_max)
        crossbar = convert_range(crossbar, weight_min, weight_max, 1 / r_off, 1 / r_on)
        return crossbar
    else:
        raise ("%s is not currently supported." % scheme)
