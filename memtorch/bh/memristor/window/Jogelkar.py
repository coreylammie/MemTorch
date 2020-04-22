import torch
import memtorch
import numpy as np
import math


def Jogelkar(x, p):
    """Jogelkar window function.

    Parameters
    ----------
    x : float
        State variable.
    p : int
        p constant.
    """
    return (1 - (2 * x - 1)) ** (2 * p)
