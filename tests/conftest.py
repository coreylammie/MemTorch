import inspect

import pytest
import torch

import memtorch
from memtorch.bh.crossbar.Program import naive_program
from memtorch.map.Parameter import naive_map
from memtorch.mn.Module import patch_model, supported_module_parameters


@pytest.fixture
def debug_networks():
    default_kwargs = {
        "in_features": 2,
        "out_features": 2,
        "in_channels": 1,
        "out_channels": 2,
        "kernel_size": 1,
        "padding": 1,
        "bias": True,
    }
    networks = []
    device = torch.device("cpu" if "cpu" in memtorch.__version__ else "cuda")
    for supported_module_parameter in supported_module_parameters:

        class Network(torch.nn.Module):
            def __init__(self):
                super(Network, self).__init__()
                layer_type = supported_module_parameters[
                    supported_module_parameter
                ].__bases__[0]
                layer_args = list(inspect.signature(layer_type.__init__).parameters)
                args = {}
                layer_args.pop(0)
                for layer_arg in layer_args:
                    if layer_arg in default_kwargs:
                        args[layer_arg] = default_kwargs[layer_arg]

                self.layer = layer_type(**args)

            def forward(self, input):
                return self.layer(input)

        networks.append(Network().to(device))

    return networks


@pytest.fixture
def debug_patched_networks(debug_networks):
    def debug_patched_networks_(tile_shape, quant_method):
        if quant_method is not None:
            ADC_resolution = 8
        else:
            ADC_resolution = None

        networks = debug_networks
        device = torch.device("cpu" if "cpu" in memtorch.__version__ else "cuda")
        patched_networks = []
        for network in networks:
            patched_networks.append(
                patch_model(
                    network,
                    memristor_model=memtorch.bh.memristor.LinearIonDrift,
                    memristor_model_params={"time_series_resolution": 0.1},
                    module_parameters_to_patch=[type(network.layer)],
                    mapping_routine=naive_map,
                    transistor=True,
                    programming_routine=None,
                    scheme=memtorch.bh.Scheme.SingleColumn,
                    tile_shape=tile_shape,
                    max_input_voltage=1.0,
                    ADC_resolution=ADC_resolution,
                    quant_method=quant_method,
                )
            )

        return patched_networks

    return debug_patched_networks_
