import time

import torch

import memtorch
import memtorch_bindings
import memtorch_cuda_bindings
from memtorch.bh.crossbar.Tile import gen_tiles
import copy

tensor_shape = (2, 10)
tensor = torch.zeros(tensor_shape).uniform_(0, 1)
bits = 4
overflow_rate = 0.5

print(tensor)
# Linear
python_output = copy.deepcopy(tensor)
memtorch_bindings.quantize(
    python_output, bits, overflow_rate, quant_method=0, min=0.1, max=0.9)
print(python_output)
cpp_output = memtorch_cuda_bindings.quantize(
    tensor, bits, overflow_rate, quant_method=0, min=0.1, max=0.9)
print(cpp_output)
# Log
# TBD ....
# Tanh
# TBD ....
