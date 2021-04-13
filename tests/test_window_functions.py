import copy
import math
import random

import numpy as np
import pytest
import torch

import memtorch


@pytest.mark.parametrize("x", [0, 0.25, 0.5, 0.75, 1.0])
def test_window_functions(x):
    for method in dir(memtorch.bh.memristor.window):
        if "__" in method:
            break

        method = getattr(memtorch.bh.memristor.window, method)
        assert callable(method)
        assert type(method(x)) == float
