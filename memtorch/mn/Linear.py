import torch
import torch.nn as nn
import memtorch
from memtorch.bh.crossbar.Crossbar import init_crossbar
from memtorch.bh.crossbar.Crossbar import simulate
from memtorch.utils import convert_range
from memtorch.map.Module import naive_tune
from memtorch.map.Parameter import naive_map
import numpy as np


class Linear(nn.Linear):
    """nn.Linear equivalent.

    Parameters
    ----------
    linear_layer : torch.nn.Linear
        Linear layer to patch.
    memristor_model : memtorch.bh.memristor.Memristor.Memristor
        Memristor model.
    memristor_model_params : **kwargs
        Memristor model keyword arguments.
    mapping_routine : function
        Mapping routine to use.
    transistor : bool
        Used to determine if a 1T1R (True) or 1R arrangement (False) is simulated.
    programming_routine : function
        Programming routine to use.
    scheme : memtorch.bh.Scheme
        Weight representation scheme.
    """

    def __init__(self, linear_layer, memristor_model, memristor_model_params, mapping_routine=naive_map, transistor=True, programming_routine=None, scheme=memtorch.bh.Scheme.DoubleColumn, **kwargs):
        assert isinstance(linear_layer, nn.Linear), 'linear_layer is not an instance of nn.Linear.'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        super(Linear, self).__init__(linear_layer.in_features, linear_layer.out_features, **kwargs)
        self.weight.data = linear_layer.weight.data
        if linear_layer.bias is not None:
            self.bias.data = linear_layer.bias.data
        else:
            self.bias = None

        self.zero_grad()
        self.weight.requires_grad = False
        if linear_layer.bias is not None:
            self.bias.requires_grad = False

        self.crossbars, self.crossbar_operation = init_crossbar(weights=self.weight,
                                                               memristor_model=memristor_model,
                                                               memristor_model_params=memristor_model_params,
                                                               transistor=transistor,
                                                               mapping_routine=mapping_routine,
                                                               programming_routine=programming_routine,
                                                               scheme=scheme)
        self.transform_output = lambda x: x
        print('Patched %s -> %s' % (linear_layer, self))

    def forward(self, input):
        """Method to perform forward propagations.

            Parameters
            ----------
            input : torch.Tensor
                Input tensor.

            Returns
            -------
            torch.Tensor
                Output tensor.
        """
        if hasattr(self, 'non_linear'):
            input = convert_range(input, input.min(), input.max(), -1, 1)
            input = input.cpu().detach().numpy()
            if hasattr(self, 'simulate'):
                out = torch.tensor(self.transform_output(self.crossbar_operation(self.crossbars, lambda crossbar, input: simulate(input, crossbar.devices, nl=False), input))).to(self.device)
            else:
                out = torch.tensor(self.transform_output(self.crossbar_operation(self.crossbars, lambda crossbar, input: simulate(input, crossbar.devices, nl=True), input))).to(self.device)
        else:
            out = torch.matmul(input.to(self.device), self.crossbar_operation(self.crossbars, lambda crossbar: crossbar.conductance_matrix))

        out =  self.transform_output(out)
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out)

        return out


    def forward_legacy(self, input):
        """Legacy method to perform forward propagations.

            Parameters
            ----------
            input : torch.Tensor
                Input tensor.

            Returns
            -------
            torch.Tensor
                Output tensor.
        """
        out = torch.matmul(input.to(self.device), self.weight.data.T.to(self.device))
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out)

        return out

    def tune(self):
        """Tuning method."""
        self.transform_output = naive_tune(self, (4098, self.in_features))

    def __str__(self):
        return "bh.Linear(in_features=%d, out_features=%d, bias=%s)" % (self.in_features, self.out_features, not self.bias is None)
