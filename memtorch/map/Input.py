import numpy as np
import torch
import torch.functional as F
import torch.nn as nn

import memtorch
from memtorch.utils import convert_range


def naive_scale(module, input):
    if module.max_input_voltage is not None:
        assert (
            type(module.max_input_voltage) == int
            or type(module.max_input_voltage) == float
        ) and module.max_input_voltage > 0, (
            "The maximum input voltage (max_input_voltage) must be >0."
        )
        input_range = torch.amax(torch.abs(input))
        input = convert_range(
            input,
            -input_range,
            input_range,
            -module.max_input_voltage,
            module.max_input_voltage,
        )

    return input
