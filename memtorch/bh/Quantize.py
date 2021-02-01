<<<<<<< HEAD
# Wrapper for the pytorch-playground quant.py script
=======
# Wrapper for pytorch-playground quantization script
>>>>>>> Resolved quant.py Import
import importlib
utee = importlib.import_module('.utee', 'memtorch.submodules.pytorch-playground')
import torch
import numpy as np

<<<<<<< HEAD
def quantize(input, bits, overflow_rate, quant_method='linear'):
    quant_methods = ['linear', 'log', 'log_minmax', 'minmax', 'tanh']
    assert quant_method in quant_methods, 'quant_method is not valid.'
    if quant_method == 'linear':
        sf = bits - 1 - utee.compute_integral_part(input, overflow_rate)
        return utee.linear_quantize(input, sf, bits)
    elif quant_method == 'log':
        log_abs_input = torch.log(torch.abs(input))
        sf = bits - 1 - utee.compute_integral_part(log_abs_input, overflow_rate)
        return utee.log_linear_quantize(input, sf, bits)
    elif quant_method == 'log_minmax':
        return utee.log_minmax_quantize(input, bits)
    elif quant_method == 'minmax':
        return utee.min_max_quantize(input, bits)
    elif quant_method == 'tanh':
        return utee.tanh_quantize(input, bits)

if __name__ == "__main__":
    input = torch.zeros((5, 5)).uniform_(0, 1)
    print(input)
    quant_methods = ['linear', 'log', 'log_minmax', 'minmax', 'tanh']
    for quant_method in quant_methods:
        input_quantized = quantize(input, 8, 0.,  quant_method=quant_method)
        print(input_quantized)
=======
input = torch.zeros((10, 10, 10)).uniform_(0, 1)
print(input)
max_val = input.max()
max_val = float(max_val.data.cpu().numpy())
print(max_val)
utee.min_max_quantize(input, 8)
print(input)
>>>>>>> Resolved quant.py Import
