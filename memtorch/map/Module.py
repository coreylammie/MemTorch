import memtorch
from memtorch.utils import pad_tensor
import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import r2_score


def naive_tune(module, input_shape):
    """Method to determine a linear relationship between a memristive crossbar and the output for a given memristive module.

        Parameters
        ----------
        module : torch.nn.Module
            Memristive layer to tune.
        input_shape : (int, int)
            Shape of the randomly generated input used to tune a crossbar.

        Returns
        -------
        function
            Function which transforms the output of the crossbar to the expected output.
    """
    device = torch.device('cpu' if 'cpu' in memtorch.__version__ else 'cuda')
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
    transform_output = lambda x: x * reg.coef_[0] + reg.intercept_
    module.bias = tmp
    print('Tuned %s. Coefficient of determination: %f [%f, %f]' % (module, reg.score(output, legacy_output), reg.coef_[0], reg.intercept_))
    return transform_output
