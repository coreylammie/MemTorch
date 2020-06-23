import pytest
import torch
import memtorch
from memtorch.mn.Module import supported_module_parameters
from memtorch.mn.Module import patch_model
from memtorch.map.Parameter import naive_map
from memtorch.bh.crossbar.Program import naive_program
import inspect


@pytest.fixture
def debug_networks():
    default_kwargs = {'in_features': 5,
                      'out_features': 5,
                      'in_channels': 1,
                      'out_channels': 3,
                      'kernel_size': 3,
                      'bias': True}
    networks = []
    device = torch.device('cpu' if 'cpu' in memtorch.__version__ else 'cuda')
    for supported_module_parameter in supported_module_parameters:
        class Network(torch.nn.Module):
            def __init__(self):
                super(Network, self).__init__()
                layer_type = supported_module_parameters[supported_module_parameter].__bases__[0]
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
    networks = debug_networks
    device = torch.device('cpu' if 'cpu' in memtorch.__version__ else 'cuda')
    patched_networks = []
    for network in networks:
        patched_networks.append(patch_model(network,
                                      memristor_model=memtorch.bh.memristor.LinearIonDrift,
                                      memristor_model_params={},
                                      module_parameters_to_patch=[type(network.layer)],
                                      mapping_routine=naive_map,
                                      transistor=True,
                                      programming_routine=None,
                                      scheme=memtorch.bh.Scheme.SingleColumn))

    return patched_networks
