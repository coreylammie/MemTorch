import torch
import memtorch
import numpy as np
import math


def Prodromakis(x, p=1, j=1.0):
    """Prodromakis window function.

    Parameters
    ----------
    x : float
        State variable.
    p : int
        p constant.
    j : float
        j constant.
    """
    return j * ((1 - (x - 0.5) ** 2 + 0.75) ** p)
