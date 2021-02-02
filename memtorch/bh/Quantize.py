# Wrapper for the pytorch-playground quant.py script
import importlib
utee = importlib.import_module('.utee', 'memtorch.submodules.pytorch-playground')
import torch
import numpy as np
quant_methods = ['linear', 'log', 'tanh']

def quantize(input, bits, overflow_rate, quant_method='linear'):
    """Method to quantize a tensor.

    Parameters
    ----------
    input : tensor
        Input tensor.
    bits : int
        Bit width.
    overflow_rate : float
        Overflow rate threshold for linear quanitzation.
    quant_method : str
        Quantization method. Must be in ['linear', 'log', 'tanh'].

    Returns
    -------
    tensor
        Quantized tensor.

    """
    assert type(bits) == int and bits > 0, 'bits must be an integer > 0.'
    assert overflow_rate >= 0 and overflow_rate <= 1, 'overflow_rate value invalid.'
    assert quant_method in quant_methods, 'quant_method is not valid.'
    if quant_method == 'linear':
        sf = bits - 1 - utee.compute_integral_part(input, overflow_rate)
        return utee.linear_quantize(input, sf, bits)
    elif quant_method == 'log':
        log_abs_input = torch.log(torch.abs(input))
        sf = bits - 1 - utee.compute_integral_part(log_abs_input, overflow_rate)
        return utee.log_linear_quantize(input, sf, bits)
    elif quant_method == 'tanh':
        return utee.tanh_quantize(input, bits)
