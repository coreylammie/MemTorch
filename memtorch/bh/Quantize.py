# Wrapper for pytorch-playground quantization script
import importlib
utee = importlib.import_module('.utee', 'memtorch.submodules.pytorch-playground')
import torch
import numpy as np

input = torch.zeros((10, 10, 10)).uniform_(0, 1)
print(input)
max_val = input.max()
max_val = float(max_val.data.cpu().numpy())
print(max_val)
utee.min_max_quantize(input, 8)
print(input)
