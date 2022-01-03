import copy
import inspect
import math

import numpy as np
import torch
import memtorch
from memtorch.bh.crossbar.Program import naive_program
from memtorch.map.Parameter import naive_map
from memtorch.mn.Module import patch_model, supported_module_parameters
from timeit import default_timer as timer

def test():
    default_kwargs = {
        "in_features": 2,
        "out_features": 2,
        "in_channels": 1,
        "out_channels": 2,
        "kernel_size": 1,
        "padding": 1,
        "bias": True,
    }
    device = torch.device("cpu" if "cpu" in memtorch.__version__ else "cuda")
    networks =[]
    for s_m_p in supported_module_parameters:
        class Network(torch.nn.Module):
            def __init__(self):
                super(Network, self).__init__()
                layer_type = supported_module_parameters[
                    s_m_p
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
    voltage = -1.4
    p_width = 6e-7
    accuracies = []
    voltages = [-1.1, -1.2, -1.3, -1.4, -1.5, -2]
    pulse_widths = [2e-7, 4e-7, 6e-7, 8e-7, 1e-6]
    new_gs = []
    gs = []
    """
    for voltage in voltages:
        for p_width in pulse_widths:
            time_signal, neg_voltage_signal = gen_programming_signal(
                1,
                p_width,
                0,
                voltage,
                1e-10,
            )
            device = Data_Driven2021()
            R0 = 1 / device.g
            g_0 = device.g
            for i in range(1):
                device.simulate(neg_voltage_signal)
            good_g = device.g
            N = 2000
            dx = (p_width - 0) / N
            t_midpoint = np.linspace(dx / 2, p_width - dx / 2, N)
            r_n = device.r_n[0] + device.r_n[1] * voltage
            s_n = device.A_n * (math.exp(abs(voltage) / device.t_n) - 1)

            def f(t_2, R0, s_n, r_n):
                return (s_n * r_n * (r_n - (R0)) * t_2 + R0) / (1 + s_n * (r_n - R0) * t_2)

            # f = lambda t_2: (s_n * r_n * (r_n - (R0)) * t_2 + R0) / (1 + s_n * (r_n - R0)*t_2)

            for i in range(1):
                newR = f(p_width, R0, s_n, r_n)
                R0 = newR
            new_g_diff = g_0 - 1/R0
            new_gs.append(new_g_diff)
            gs.append(g_0 - good_g)
            accuracies.append(abs((new_g_diff - (g_0 - good_g)) / (g_0 - good_g) * 100))
    print()
    print(accuracies)
    print(new_gs)
    print(gs)
    print()
    print("Mean accuracy = " + str(100 - np.mean(accuracies)))

    return
    """
    for network in networks:
        print()
        print("-------------------- New network --------------------")
        print(type(network.layer))
        print("Starting optimized routine")
        start = timer()
        patched_network = patch_model(
            copy.deepcopy(network),
            memristor_model=memtorch.bh.memristor.Data_Driven2021,
            memristor_model_params={"r_off" : 6000},
            module_parameters_to_patch=[type(network.layer)],
            mapping_routine=naive_map,
            transistor=False,
            programming_routine=naive_program,
            scheme=memtorch.bh.Scheme.DoubleColumn,
            programming_routine_params={"rel_tol": 0.01,
                                        "pulse_duration": 2e-7,
                                        "refactory_period": 0,
                                        "pos_voltage_level": 1.0,
                                        "neg_voltage_level": -1.0,
                                        "timeout": 5,
                                        "simulate_neighbours" : False,
                                        "force_adjustment": 1e-2,
                                        "force_adjustment_rel_tol": 1e-1,
                                        "force_adjustment_pos_voltage_threshold": 1.1,
                                        "force_adjustment_neg_voltage_threshold": -1.5, },
            tile_shape=(128,128),
            max_input_voltage=1.0,
            ADC_resolution=None,
            quant_method=None,
            source_resistance=5,
            line_resistance=5,
            random_crossbar_init=True,
            use_bindings=True,
        )
        end = timer()
        print("Optimized routine time took: " + str(end - start))
        ##
        """
        start = timer()
        patched_network = patch_model(
            copy.deepcopy(network),
            memristor_model=memtorch.bh.memristor.Data_Driven2021,
            memristor_model_params={"r_off": 6000, "time_series_resolution": 2e-7},
            module_parameters_to_patch=[type(network.layer)],
            mapping_routine=naive_map,
            transistor=False,
            programming_routine=naive_program,
            scheme=memtorch.bh.Scheme.DoubleColumn,
            programming_routine_params={"rel_tol": 0.01,
                                        "pulse_duration": 2e-7,
                                        "refactory_period": 0,
                                        "pos_voltage_level": 1.0,
                                        "neg_voltage_level": -1.0,
                                        "timeout": 5,
                                        "force_adjustment": 1e-2,
                                        "force_adjustment_rel_tol": 1e-1,
                                        "force_adjustment_pos_voltage_threshold": 1.1,
                                        "force_adjustment_neg_voltage_threshold": -1.5, },
            tile_shape=(24, 24),
            max_input_voltage=1.0,
            ADC_resolution=None,
            quant_method=None,
            source_resistance=5,
            line_resistance=5,
            random_crossbar_init=True,
            use_bindings=False,
        )
        end = timer()
        print("Old routine time took: " + str(end - start))
        #patched_network.tune_()
        patched_network.disable_legacy()
        """

if __name__ == "__main__":
    test()