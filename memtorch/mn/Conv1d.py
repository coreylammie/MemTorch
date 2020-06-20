import torch
import torch.nn as nn
import memtorch
from memtorch.bh.crossbar.Crossbar import init_crossbar
from memtorch.utils import convert_range
from memtorch.map.Module import naive_tune
from memtorch.map.Parameter import naive_map
import numpy as np


class Conv1d(nn.Conv1d):
    """nn.Conv1d equivalent.

    Parameters
    ----------
    convolutional_layer : torch.nn.Conv1d
        Convolutional layer to patch.
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
    p_l: float
        If not None, the proportion of weights to retain.
    scheme : memtorch.bh.Scheme
        Weight representation scheme.
    """

    def __init__(self, convolutional_layer, memristor_model, memristor_model_params, mapping_routine=naive_map, transistor=False, programming_routine=None, p_l=None, scheme=memtorch.bh.Scheme.DoubleColumn, *args, **kwargs):
        assert isinstance(convolutional_layer, nn.Conv1d), 'convolutional_layer is not an instance of nn.Conv1d.'
        self.device = torch.device('cpu' if 'cpu' in memtorch.__version__ else 'cuda')
        super(Conv1d, self).__init__(convolutional_layer.in_channels, convolutional_layer.out_channels, convolutional_layer.kernel_size, **kwargs)
        self.padding = convolutional_layer.padding
        self.stride = convolutional_layer.stride
        self.weight.data = convolutional_layer.weight.data
        if convolutional_layer.bias is not None:
            self.bias.data = convolutional_layer.bias.data

        self.zero_grad()
        self.weight.requires_grad = False
        if convolutional_layer.bias is not None:
            self.bias.requires_grad = False

        self.crossbars, self.crossbar_operation = init_crossbar(weights=self.weight,
                                                               memristor_model=memristor_model,
                                                               memristor_model_params=memristor_model_params,
                                                               transistor=transistor,
                                                               mapping_routine=mapping_routine,
                                                               programming_routine=programming_routine,
                                                               p_l=p_l,
                                                               scheme=scheme)
        self.transform_output = lambda x: x
        print('Patched %s -> %s' % (convolutional_layer, self))


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
        if self.forward_legacy_enabled:
            return torch.nn.functional.conv1d(input.to(self.device), self.weight, bias=self.bias, stride=self.stride, padding=self.padding)
        else:
            output_dim = int((input.shape[2] - self.kernel_size[0] + 2 * self.padding[0]) / self.stride[0]) + 1
            out = torch.zeros((input.shape[0], self.out_channels, output_dim)).to(self.device)
            if hasattr(self, 'non_linear'):
                input = convert_range(input, input.min(), input.max(), -1, 1)
            else:
                weight = self.crossbar_operation(self.crossbars, lambda crossbar: crossbar.conductance_matrix).view(self.weight.shape)

            for batch in range(input.shape[0]):
                filter = torch.zeros((self.in_channels, self.kernel_size[0]))
                count = 0
                for i in range(self.out_channels):
                    while count < (input.shape[-1] - self.kernel_size[0] + 1):
                        for j in range(self.in_channels):
                            for k in range(count, self.kernel_size[0] + count):
                                if hasattr(self, 'non_linear') and hasattr(self, 'simulate'):
                                    out[batch][i][count] = out[batch][i][count] + self.crossbar_operation(self.crossbars, lambda crossbar: crossbar.devices[i][j][k - count].simulate(input[batch][j][k], return_current=True)).item()
                                elif hasattr(self, 'non_linear'):
                                    out[batch][i][count] = out[batch][i][count] + self.crossbar_operation(self.crossbars, lambda crossbar: crossbar.devices[i][j][k - count].det_current(input[batch][j][k])).item()
                                else:
                                    out[batch][i][count] = out[batch][i][count] + (input[batch][j][k] * weight[i][j][k - count].item())

                        count = count + 1
                    count = 0

            out = self.transform_output(out)
            if self.bias is not None:
                out += self.bias.view(-1, 1).expand_as(out)

            return out

    def tune(self):
        """Tuning method."""
        self.transform_output = naive_tune(self, (8, self.in_channels, 32))

    def __str__(self):
        return "bh.Conv1d(in_channels=%d, out_channels=%d, kernel_size=%d, stride=%d, padding=%d)" % (self.in_channels, self.out_channels, self.kernel_size[0], self.stride[0], self.padding[0])
