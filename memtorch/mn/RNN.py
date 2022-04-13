import math
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules import conv

import memtorch
from memtorch.bh.crossbar.Crossbar import init_crossbar, simulate_matmul
from memtorch.bh.crossbar.Tile import tiled_inference
from memtorch.map.Input import naive_scale
from memtorch.map.Module import naive_tune
from memtorch.map.Parameter import naive_map


class RNN(nn.RNN):
    """nn.RNN equivalent.

    Parameters
    ----------
    rnn_layer : torch.nn.RNN
        RNN layer to patch.
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
    max_input_voltage : float
        Maximum input voltage used to encode inputs. If None, inputs are unbounded.
    scaling_routine : function
        Scaling routine to use in order to scale batch inputs.
    scaling_routine_params : **kwargs
        Scaling routine keyword arguments.
    source_resistance : float
        The resistance between word/bit line voltage sources and crossbar(s).
    line_resistance : float
        The interconnect line resistance between adjacent cells.
    ADC_resolution : int
        ADC resolution (bit width). If None, quantization noise is not accounted for.
    ADC_overflow_rate : float
        Overflow rate threshold for linear quanitzation (if ADC_resolution is not None).
    quant_method: string
        Quantization method. Must be in ['linear', 'log', 'log_minmax', 'minmax', 'tanh'], or None.
    use_bindings : bool
        Used to determine if C++/CUDA bindings are used (True) or not (False).
    random_crossbar_init: bool
        Determines if the crossbar is to be initialized at random values in between Ron and Roff
    verbose : bool
        Used to determine if verbose output is enabled (True) or disabled (False).
    """

    def __init__(
        self,
        rnn_layer,
        memristor_model,
        memristor_model_params,
        mapping_routine=naive_map,
        transistor=True,
        programming_routine=None,
        programming_routine_params={},
        p_l=None,
        scheme=memtorch.bh.Scheme.DoubleColumn,
        tile_shape=None,
        max_input_voltage=None,
        scaling_routine=naive_scale,
        scaling_routine_params={},
        source_resistance=None,
        line_resistance=None,
        ADC_resolution=None,
        ADC_overflow_rate=0.0,
        quant_method=None,
        use_bindings=True,
        random_crossbar_init=False,
        verbose=True,
        *args,
        **kwargs,
    ):
        assert isinstance(rnn_layer, nn.RNN), "rnn_layer is not an instance of nn.RNN."
        self.device = torch.device("cpu" if "cpu" in memtorch.__version__ else "cuda")
        self.transistor = transistor
        self.scheme = scheme
        self.tile_shape = tile_shape
        self.max_input_voltage = max_input_voltage
        self.scaling_routine = scaling_routine
        self.scaling_routine_params = scaling_routine_params
        self.source_resistance = source_resistance
        self.line_resistance = line_resistance
        self.ADC_resolution = ADC_resolution
        self.ADC_overflow_rate = ADC_overflow_rate
        if "cpu" not in memtorch.__version__:
            self.cuda_malloc_heap_size = 50
        else:
            self.cuda_malloc_heap_size = None

        if not transistor:
            assert (
                source_resistance is not None and source_resistance >= 0.0
            ), "Source resistance is invalid."
            assert (
                line_resistance is not None and line_resistance >= 0.0
            ), "Line resistance is invalid."

        if quant_method in memtorch.bh.Quantize.quant_methods:
            self.quant_method = quant_method
        else:
            self.quant_method = None

        if quant_method is not None:
            assert (
                ADC_resolution is not None
                and type(ADC_resolution) == int
                and ADC_resolution > 0
            ), "ADC resolution is invalid."
            assert (
                ADC_overflow_rate is not None
            ), "ADC_overflow_rate must be specified if quant_method is not None."

        self.use_bindings = use_bindings
        self.verbose = verbose
        self.forward_legacy_enabled = True
        super(RNN, self).__init__(
            input_size=rnn_layer.input_size,
            hidden_size=rnn_layer.hidden_size,
            num_layers=rnn_layer.num_layers,
            nonlinearity=rnn_layer.nonlinearity,
            bias=rnn_layer.bias,
            batch_first=False,  # To add support.
            dropout=0.0,  # To add support.
            bidirectional=rnn_layer.bidirectional,
            **kwargs,
        )
        if rnn_layer.nonlinearity in ["tanh", "relu"]:
            if rnn_layer.nonlinearity == "tanh":
                self.nonlinearity = torch.tanh
            elif rnn_layer.nonlinearity == "relu":
                self.nonlinearity = torch.relu
        else:
            raise Exception("Nonlinearity must be either tanh or relu")

        self.w_ih = []
        self.w_hh = []
        if rnn_layer.bias:
            self.b_ih = []
            self.b_hh = []

        if rnn_layer.bidirectional:
            self.w_ih_reverse = []
            self.w_hh_reverse = []
            if rnn_layer.bias:
                self.b_ih_reverse = []
                self.b_hh_reverse = []

        self.zero_grad()
        for i in range(rnn_layer.num_layers):
            self.w_ih.append(rnn_layer._parameters[f"weight_ih_l{i}"].data)
            self.w_ih[i].requires_grad = False
            self.w_hh.append(rnn_layer._parameters[f"weight_hh_l{i}"].data)
            self.w_hh[i].requires_grad = False
            if rnn_layer.bias:
                self.b_ih.append(rnn_layer._parameters[f"bias_ih_l{i}"].data)
                self.b_ih[i].requires_grad = False
                self.b_hh.append(rnn_layer._parameters[f"bias_hh_l{i}"].data)
                self.b_hh[i].requires_grad = False

            if rnn_layer.bidirectional:
                self.w_ih_reverse.append(
                    rnn_layer._parameters["weight_ih_l{i}_reverse"].data
                )
                self.w_ih_reverse[i].requires_grad = False
                self.w_hh_reverse.append(
                    rnn_layer._parameters["weight_hh_l{i}_reverse"].data
                )
                self.w_hh_reverse[i].requires_grad = False
                if rnn_layer.bias:
                    self.b_ih_reverse.append(
                        rnn_layer._parameters["bias_ih_l{i}_reverse"].data
                    )
                    self.b_ih_reverse[i].requires_grad = False
                    self.b_hh_reverse.append(
                        rnn_layer._parameters["bias_hh_l{i}_reverse"].data
                    )
                    self.b_hh_reverse[i].requires_grad = False

        warnings.warn(
            "RNN layers are not fully supported. Only legacy forward passes are currently enabled."
        )

    def forward(self, input, h_0=None):
        """Method to perform forward propagations.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor.

                h_0 : torch.Tensor
                        The initial hidden state for the input sequence batch

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        if h_0 is None:
            if self.bidirectional:
                h_0 = torch.zeros(2, self.num_layers, input.shape[1], self.hidden_size)
            else:
                h_0 = torch.zeros(1, self.num_layers, input.shape[1], self.hidden_size)

        if self.bidirectional:
            output = torch.zeros(input.shape[0], input.shape[1], 2 * self.hidden_size)
        else:
            output = torch.zeros(input.shape[0], input.shape[1], self.hidden_size)

        inp = input
        for layer in range(self.num_layers):
            h_t = h_0[0, layer]
            for t in range(inp.shape[0]):
                if self.bias:
                    h_t = (
                        torch.matmul(inp[t], self.w_ih[layer].T)
                        + self.b_ih[layer]
                        + torch.matmul(h_t, self.w_hh[layer].T)
                        + self.b_hh[layer]
                    )
                else:
                    h_t = torch.matmul(inp[t], self.w_ih[layer].T) + torch.matmul(
                        h_t, self.w_hh[layer].T
                    )

                h_t = self.nonlinearity(h_t)
                output[t, :, : self.hidden_size] = h_t

            if self.bidirectional:
                h_t_reverse = h_0[1, layer]
                for t in range(inp.shape[0]):
                    if self.bias:
                        h_t_reverse = (
                            torch.matmul(inp[-1 - t], self.w_ih_reverse[layer].T)
                            + self.b_ih_reverse[layer]
                            + torch.matmul(h_t_reverse, self.w_hh_reverse[layer].T)
                            + self.b_hh_reverse[layer]
                        )
                    else:
                        h_t_reverse = torch.matmul(
                            inp[-1 - t], self.w_ih_reverse[layer].T
                        ) + torch.matmul(h_t_reverse, self.w_hh_reverse[layer].T)

                    h_t_reverse = self.nonlinearity(h_t_reverse)
                    output[-1 - t, :, self.hidden_size :] = h_t_reverse

            inp = output.clone()

        return output

    def tune(self):
        """Tuning method."""
        pass  # To be implemented.

    def __str__(self):
        return (
            "bh.RNN(input_size=%d, hidden_size=%d, num_layers=%d, nonlinearity=%s, bias=%s, bidirectional=%s)"
            % (
                self.input_size,
                self.hidden_size,
                self.num_layers,
                self.nonlinearity,
                self.bias,
                self.bidirectional,
            )
        )
