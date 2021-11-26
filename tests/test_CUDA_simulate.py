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

    for network in networks:
        patched_network = patch_model(
            copy.deepcopy(network),
            memristor_model=memtorch.bh.memristor.Data_Driven2021,
            memristor_model_params={"a_n":0.666},
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
                                        "force_adjustment_neg_voltage_threshold": 0,},
            tile_shape=tile_shape,
            max_input_voltage=1.0,
            ADC_resolution=ADC_resolution,
            quant_method=quant_method,
            source_resistance=source_resistance,
            line_resistance=line_resistance,
            use_bindings=use_bindings,
        )
        #patched_network.tune_()
        patched_network.disable_legacy()
