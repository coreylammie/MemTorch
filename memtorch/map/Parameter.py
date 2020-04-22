import memtorch
from memtorch.utils import convert_range
import torch
import torch.nn as nn
import torch.functional as F
import numpy as np


def naive_map(weight, r_on, r_off, scheme):
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

        Returns
        -------
        torch.Tensor, torch.Tensor
            Positive and negative crossbar weights.
    """
    if scheme == memtorch.bh.crossbar.Scheme.DoubleColumn:
        range = weight.abs().max()
        pos = weight.clone()
        neg = weight.clone() * -1
        pos[pos < 0] = 0
        neg[neg < 0] = 0
        pos = convert_range(pos, 0, range, 1 / r_off, 1 / r_on)
        neg = convert_range(neg, 0, range, 1 / r_off, 1 / r_on)
        return pos, neg
    elif scheme == memtorch.bh.crossbar.Scheme.SingleColumn:
        range = weight.abs().max()
        crossbar = weight.clone()
        crossbar = convert_range(crossbar, crossbar.min(), crossbar.max(), 1 / r_off, 1 / r_on)
        return crossbar
    else:
        raise('%s is not currently supported.' % scheme)
