import copy

import numpy as np
import torch

import memtorch
import memtorch_bindings

quant_methods = ["linear", "log"]


def quantize(
    tensor,
    quant,
    overflow_rate=0.0,
    quant_method=None,
    min=float("nan"),
    max=float("nan"),
    override_original=False,
):
    """Method to quantize a tensor.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor.
    quant : int
        Bit width (if quant_method is not None) or the number of discrete quantization levels (if quant_method is None).
    overflow_rate : float, optional
        Overflow rate threshold for linear quantization.
    quant_method : str, optional
        Quantization method. Must be in quant_methods.
    min : float or tensor, optional
        Minimum value(s) to clip numbers to.
    max : float or tensor, optional
        Maximum value(s) to clip numbers to.
    override_original : bool, optional
        Whether to override the original tensor (True) or not (False).

    Returns
    -------
    torch.Tensor
        Quantized tensor.

    """
    device = torch.device("cpu" if "cpu" in memtorch.__version__ else "cuda")
    assert (
        overflow_rate >= 0 and overflow_rate <= 1
    ), "overflow_rate must be >= 0 and <= 1."
    assert (
        type(quant) == int and quant > 0
    ), "The bit width or number of discrete quantization levels must be a positive integer."
    if type(min) == int:
        min = float(min)
    if type(max) == int:
        max = float(max)
    if not override_original:
        tensor = copy.deepcopy(tensor)
    if quant_method is not None:
        assert quant_method in quant_methods, "quant_method is invalid."
        tensor = tensor.cpu()
        memtorch_bindings.quantize(
            tensor,
            bits=quant,
            overflow_rate=overflow_rate,
            quant_method=quant_methods.index(quant_method),
            min=min,
            max=max,
        )
    else:
        tensor = tensor.cpu()
        memtorch_bindings.quantize(tensor, n_quant_levels=quant, min=min, max=max)

    return tensor.to(device)
