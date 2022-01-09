import numpy as np
import torch
import torch.functional as F
import torch.nn as nn
from sklearn import datasets, linear_model
from sklearn.metrics import r2_score

import memtorch
from memtorch.utils import pad_tensor


def naive_tune(module, input_shape, verbose=True):
    """Method to determine a linear relationship between a memristive crossbar and the output for a given memristive module.

    Parameters
    ----------
    module : torch.nn.Module
        Memristive layer to tune.
    input_shape : int, int
        Shape of the randomly generated input used to tune a crossbar.
    verbose : bool, optional
        Used to determine if verbose output is enabled (True) or disabled (False).

    Returns
    -------
    function
        Function which transforms the output of the crossbar to the expected output.
    """
    device = torch.device("cpu" if "cpu" in memtorch.__version__ else "cuda")
    tmp = module.bias
    module.bias = None
    input = torch.rand(input_shape).uniform_(-1, 1).to(device)
    initial_forward_legacy_state = module.forward_legacy_enabled
    module.forward_legacy_enabled = False
    output = module.forward(input).detach().cpu()
    module.forward_legacy_enabled = True
    legacy_output = module.forward(input).detach().cpu()
    module.forward_legacy_enabled = initial_forward_legacy_state
    output = output.numpy().reshape(-1, 1)
    legacy_output = legacy_output.numpy().reshape(-1, 1)
    reg = linear_model.LinearRegression(fit_intercept=True).fit(output, legacy_output)
    coef = np.array(reg.coef_).item()
    intercept = np.array(reg.intercept_).item()

    def transform_output(x):
        return x * coef + intercept

    module.bias = tmp
    if verbose:
        print(
            "Tuned %s. Coefficient of determination: %f [%f, %f]"
            % (module, reg.score(output, legacy_output), coef, intercept)
        )

    return transform_output
