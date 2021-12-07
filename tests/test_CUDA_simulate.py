import copy
import math

import numpy as np
import pytest
import torch
from numpy.lib import source

import memtorch
from memtorch.bh.crossbar.Program import naive_program
from memtorch.map.Parameter import naive_map
from memtorch.mn.Module import patch_model
from memtorch.bh.memristor.Data_Driven2021 import Data_Driven2021
from memtorch.bh.crossbar.Program import gen_programming_signal


# f_2 = lambda t: (s_n*((R0-r_n)**2)*t)
# f = lambda t: (-1 + (r_n * s_n * t))/(s_n*t)

@pytest.mark.parametrize("tile_shape", [None, (128, 128), (10, 20)])
@pytest.mark.parametrize("quant_method", memtorch.bh.Quantize.quant_methods + [None])
@pytest.mark.parametrize("source_resistance", [5, 5])
@pytest.mark.parametrize("line_resistance", [5, 5])
@pytest.mark.parametrize("use_bindings", [True, False])
def test_CUDA_simulate(
        debug_networks,
        tile_shape,
        quant_method,
        source_resistance,
        line_resistance,
        use_bindings,
):
    networks = debug_networks
    if quant_method is not None:
        ADC_resolution = 8
    else:
        ADC_resolution = None
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
        print()
        patched_network = patch_model(
            copy.deepcopy(network),
            memristor_model=memtorch.bh.memristor.Data_Driven2021,
            memristor_model_params={"a_n": 0.666},
            module_parameters_to_patch=[type(network.layer)],
            mapping_routine=naive_map,
            transistor=False,
            programming_routine=naive_program,
            scheme=memtorch.bh.Scheme.SingleColumn,
            programming_routine_params={"rel_tol": 0.1,
                                        "pulse_duration": 1e-3,
                                        "refactory_period": 0,
                                        "pos_voltage_level": 1.0,
                                        "neg_voltage_level": -1.0,
                                        "timeout": 5,
                                        "force_adjustment": 1e-3,
                                        "force_adjustment_rel_tol": 1e-1,
                                        "force_adjustment_pos_voltage_threshold": 0,
                                        "force_adjustment_neg_voltage_threshold": 0, },
            tile_shape=tile_shape,
            max_input_voltage=1.0,
            ADC_resolution=ADC_resolution,
            quant_method=quant_method,
            source_resistance=source_resistance,
            line_resistance=line_resistance,
            use_bindings=use_bindings,
        )
        # patched_network.tune_()
        patched_network.disable_legacy()

def __main__():
    test_CUDA_simulate();