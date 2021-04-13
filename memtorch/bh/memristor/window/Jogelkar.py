import math

import numpy as np
import torch

import memtorch


def Jogelkar(x, p=1):
    """Jogelkar window function.

    Parameters
    ----------
    x : float
        State variable.
    p : int
        p constant.
    """
    return float(1 - (2 * x - 1)) ** (2 * p)
