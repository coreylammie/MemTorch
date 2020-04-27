import pytest
import torch
if torch.cuda.is_available():
    import cuda_quantization as quantization
else:
    import quantization

import memtorch
import matplotlib
import math
import numpy as np
import random
import bisect
import copy


@pytest.mark.parametrize('shape, quantization_levels', [((20, 50), 10), ((100, 100), 5)])
def test_quantize(shape, quantization_levels):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tensor = torch.zeros(shape).uniform_(0, 1).to(device)
    quantized_tensor = copy.deepcopy(tensor)
    quantization.quantize(quantized_tensor, quantization_levels, 0, 1)
    valid_values = torch.linspace(0, 1, quantization_levels)
    quantized_tensor_unique = quantized_tensor.unique()
    assert any([bool(val) for val in [torch.isclose(quantized_tensor_unique, valid_value).any() for valid_value in valid_values]])
    assert tensor.shape == quantized_tensor.shape
    assert math.isclose(min(valid_values.tolist(), key=lambda x: abs(x - tensor[0][0])), quantized_tensor[0][0])
