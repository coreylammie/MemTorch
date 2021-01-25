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


class Conv2d(nn.Conv2d):
    """nn.Conv2d equivalent.

    Parameters
    ----------
    convolutional_layer : torch.nn.Conv2d
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
    programming_routine_params : **kwargs
        Programming routine keyword arguments.
    p_l: float
        If not None, the proportion of weights to retain.
    scheme : memtorch.bh.Scheme
        Weight representation scheme.
    tile_shape : (int, int)
        Tile shape to use to store weights. If None, modular tiles are not used.
    """

    def __init__(self, convolutional_layer, memristor_model, memristor_model_params, mapping_routine=naive_map, transistor=True, programming_routine=None, programming_routine_params={}, p_l=None, scheme=memtorch.bh.Scheme.DoubleColumn, tile_shape=(128, 128), *args, **kwargs):
        assert isinstance(convolutional_layer, nn.Conv2d), 'convolutional_layer is not an instance of nn.Conv2d.'
        self.device = torch.device('cpu' if 'cpu' in memtorch.__version__ else 'cuda')
        self.scheme = scheme
        self.tile_shape = tile_shape
        self.forward_legacy_enabled = True
        super(Conv2d, self).__init__(convolutional_layer.in_channels, convolutional_layer.out_channels, convolutional_layer.kernel_size, **kwargs)
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
                                                               programming_routine_params=programming_routine_params,
                                                               p_l=p_l,
                                                               scheme=scheme,
                                                               tile_shape=tile_shape)
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
            return torch.nn.functional.conv2d(input.to(self.device), self.weight.to(self.device), bias=self.bias, stride=self.stride, padding=self.padding)
        else:
            output_dim = [0, 0]
            output_dim[0] = int((input.shape[2] - self.kernel_size[0] + 2 * self.padding[0]) / self.stride[0]) + 1
            output_dim[1] = int((input.shape[3] - self.kernel_size[1] + 2 * self.padding[1]) / self.stride[1]) + 1
            out = torch.zeros((input.shape[0], self.out_channels, output_dim[0], output_dim[1])).to(self.device)
            for batch in range(input.shape[0]):
                unfolded_batch_input = torch.nn.functional.unfold(input[batch, :, :, :].unsqueeze(0), kernel_size=self.kernel_size, stride=self.stride, padding=self.padding).squeeze().T
                unfolded_batch_input_shape = unfolded_batch_input.shape
                if hasattr(self, 'non_linear'):
                    raise Exception('TBD.') # TODO
                    # unfolded_batch_input = convert_range(unfolded_batch_input, unfolded_batch_input.min(), unfolded_batch_input.max(), -1, 1) #.squeeze(0)
                    # unfolded_batch_input = unfolded_batch_input.transpose(1, 0).cpu().detach().numpy()
                    # if hasattr(self, 'simulate'):
                    #     out_ = self.transform_output(self.crossbar_operation(self.crossbars, lambda crossbar, input_: simulate_matmul(input_, crossbar.devices.transpose(1, 0), nl=False), input_=unfolded_batch_input)).to(self.device).T.unsqueeze(0)
                    # else:
                    #     out_ = self.transform_output(self.crossbar_operation(self.crossbars, lambda crossbar, input_: simulate_matmul(input_, crossbar.devices.transpose(1,0), nl=True), input_=unfolded_batch_input)).to(self.device).T.unsqueeze(0)
                else:
                    if self.tile_shape is not None:
                        unfolded_batch_input_tiles, unfolded_batch_input_tiles_map = gen_tiles(unfolded_batch_input, self.tile_shape, input=True)
                        crossbar_shape = (self.crossbars[0].rows, self.crossbars[0].columns)
                        tiles_map = self.crossbars[0].tiles_map
                        out_ = tile_matmul(unfolded_batch_input_tiles, unfolded_batch_input_tiles_map, unfolded_batch_input_shape, self.crossbar_operation(self.crossbars, lambda crossbar: crossbar.conductance_matrix), tiles_map, crossbar_shape).T
                    else:
                        out_ = self.transform_output(torch.matmul(unfolded_batch_input, self.crossbar_operation(self.crossbars, lambda crossbar: crossbar.conductance_matrix))).T

                out[batch] = out_.view(size=(1, self.out_channels, output_dim[0], output_dim[1]))

            if self.bias is not None:
                out += self.bias.data.view(-1, 1, 1).expand_as(out)

            return out

    def tune(self, input_batch_size=8, input_shape=32):
        """Tuning method."""
        self.transform_output = naive_tune(self, (input_batch_size, self.in_channels, input_shape, input_shape))

    def __str__(self):
        return "bh.Conv2d(in_channels=%d, out_channels=%d, kernel_size=(%d, %d), stride=(%d, %d), padding=(%d, %d))" % (self.in_channels, self.out_channels, self.kernel_size[0], self.kernel_size[1], self.stride[0], self.stride[1], self.padding[0], self.padding[1])
