import numpy as np
import torch
import torch.functional as F
import torch.nn as nn

import memtorch
from memtorch.utils import convert_range


def naive_scale(module, input, force_scale=False):
    """Naive method to encode input values as bit-line voltages.

    Parameters
    ----------
    module : torch.nn.Module
        Memristive layer to tune.
    input : torch.tensor
        Input tensor to encode.
    force_scale : bool, optional
        Used to determine if inputs are scaled (True) or not (False) if they no not exceed max_input_voltage.

    Returns
    -------
    torch.Tensor
        Encoded voltages.
    """
    if module.max_input_voltage is not None:
        assert (
            type(module.max_input_voltage) == int
            or type(module.max_input_voltage) == float
        ) and module.max_input_voltage > 0, (
            "The maximum input voltage (max_input_voltage) must be >0."
        )
        input_range = torch.amax(torch.abs(input))
        if not force_scale and input_range <= module.max_input_voltage:
            return input
        else:
            return convert_range(
                input,
                -input_range,
                input_range,
                -module.max_input_voltage,
                module.max_input_voltage,
            )

    return input
