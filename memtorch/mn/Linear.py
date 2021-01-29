import torch
import torch.nn as nn
import memtorch
from memtorch.bh.crossbar.Crossbar import init_crossbar, simulate_matmul
from memtorch.bh.crossbar.Tile import gen_tiles, tile_matmul
from memtorch.utils import convert_range, pad_tensor
from memtorch.map.Module import naive_tune
from memtorch.map.Parameter import naive_map
import numpy as np
import math


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
    programming_routine_params : **kwargs
        Programming routine keyword arguments.
    p_l: float
        If not None, the proportion of weights to retain.
    scheme : memtorch.bh.Scheme
        Weight representation scheme.
    tile_shape : (int, int)
        Tile shape to use to store weights. If None, modular tiles are not used.
    """

    def __init__(self, linear_layer, memristor_model, memristor_model_params, mapping_routine=naive_map, transistor=True, programming_routine=None, programming_routine_params={}, p_l=None, scheme=memtorch.bh.Scheme.DoubleColumn, tile_shape=(128, 128), **kwargs):
        assert isinstance(linear_layer, nn.Linear), 'linear_layer is not an instance of nn.Linear.'
        self.device = torch.device('cpu' if 'cpu' in memtorch.__version__ else 'cuda')
        self.scheme = scheme
        self.tile_shape = tile_shape
        self.forward_legacy_enabled = True
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
                                                               programming_routine_params=programming_routine_params,
                                                               p_l=p_l,
                                                               scheme=scheme,
                                                               tile_shape=tile_shape)
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
        if self.forward_legacy_enabled:
            out = torch.matmul(input.to(self.device), self.weight.data.T.to(self.device))
            if self.bias is not None:
                out += self.bias.view(1, -1).expand_as(out)

            return out
        else:
            input_shape = input.shape
            if hasattr(self, 'non_linear'):
                input = convert_range(input, input.min(), input.max(), -1, 1).cpu().detach().numpy()
                if self.tile_shape is not None:
                    tiles_map = self.crossbars[0].tiles_map
                    weight_shape = self.weight.data.shape
                else:
                    tiles_map = None
                    weight_shape = None

                if hasattr(self, 'simulate'):
                    out = self.crossbar_operation(self.crossbars, lambda crossbar, input_: simulate_matmul(input, crossbar.devices, nl=False, tiles_map=tiles_map, weight_shape=weight_shape), input_=input).to(self.device)
                else:
                    out = self.crossbar_operation(self.crossbars, lambda crossbar, input_: simulate_matmul(input, crossbar.devices, nl=True, tiles_map=tiles_map, weight_shape=weight_shape), input_=input).to(self.device)
            else:
                if self.tile_shape is not None:
                    input_tiles, input_tiles_map = gen_tiles(input, self.tile_shape, input=True)
                    crossbar_shape = (self.crossbars[0].rows, self.crossbars[0].columns)
                    tiles_map = self.crossbars[0].tiles_map
                    out = tile_matmul(input_tiles, input_tiles_map, input_shape, self.crossbar_operation(self.crossbars, lambda crossbar: crossbar.conductance_matrix), tiles_map, crossbar_shape)
                else:
                    out = torch.matmul(input.to(self.device), self.crossbar_operation(self.crossbars, lambda crossbar: crossbar.conductance_matrix))

            out = self.transform_output(out)
            if self.bias is not None:
                out += self.bias.data.view(1, -1).expand_as(out)

            return out

    def tune(self, input_shape=4098):
        """Tuning method."""
        self.transform_output = naive_tune(self, (input_shape, self.in_features))

    def __str__(self):
        return "bh.Linear(in_features=%d, out_features=%d, bias=%s)" % (self.in_features, self.out_features, not self.bias is None)
