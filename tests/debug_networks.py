import torch
import memtorch
from memtorch.mn.Module import supported_module_parameters
import inspect


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
