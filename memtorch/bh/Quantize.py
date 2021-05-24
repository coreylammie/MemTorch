import memtorch
if "cpu" in memtorch.__version__:
    import memtorch_bindings
else:
    import memtorch_cuda_bindings as memtorch_bindings

import numpy as np
import torch

quant_methods = ["linear", "linear_log_bounded", "log", "tanh"]


def quantize(tensor, bits, overflow_rate=0., quant_method="linear", min=None, max=None):
    """Method to quantize a tensor.

    Parameters
    ----------
    tensor : tensor
        Input tensor.
    bits : int
        Bit width.
    overflow_rate : float, optional
        Overflow rate threshold for linear quantization.
    quant_method : str, optional
        Quantization method. Must be in ['linear', 'linear_log_bounded', 'log', 'tanh'].
    min : float or tensor, optional
        Minimum value(s) to clip numbers to.
    max : float or tensor, optional
        Maximum value(s) to clip numbers to.

    Returns
    -------
    tensor
        Quantized tensor.

    """
    assert type(bits) == int and bits > 0, "bits must be an integer > 0."
    assert overflow_rate >= 0 and overflow_rate <= 1, "overflow_rate value invalid."
    assert quant_method in quant_methods, "quant_method is not valid."
    if quant_method == "linear":
        memtorch_bindings.quantize(
            tensor, bits, min=float(min), max=float(max))
    else:
        memtorch_bindings.quantize(tensor, bits, overflow_rate=overflow_rate,
                                   quant_method=quant_methods.index(quant_method), min=float(min), max=float(max))

    return tensor
