import math

import numpy as np
import torch

import memtorch


def Biolek(voltage, x=1.0, p=1):
    """Biolek window function.

    Parameters
    ----------
    voltage : float
        Input voltage (V).
    x : float
        State variable.
    p : int
        p constant.
    """

    def step(x):
        return 1 * (x > 0)

    return 1 - (1 - x - step(voltage)) ** (2 * p)
