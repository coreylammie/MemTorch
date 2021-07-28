import copy
import math

import numpy as np
import pytest
import torch

import memtorch
from memtorch.bh.crossbar.Program import naive_program
from memtorch.map.Parameter import naive_map
from memtorch.mn.Module import patch_model


@pytest.mark.parametrize("tile_shape", [None, (128, 128), (10, 20)])
@pytest.mark.parametrize("quant_method", memtorch.bh.Quantize.quant_methods + [None])
@pytest.mark.parametrize("use_bindings", [True, False])
def test_networks(debug_networks, tile_shape, quant_method, use_bindings):
    networks = debug_networks
    if quant_method is not None:
        ADC_resolution = 8
    else:
        ADC_resolution = None

    for network in networks:
        patched_network = patch_model(
            copy.deepcopy(network),
            memristor_model=memtorch.bh.memristor.LinearIonDrift,
            memristor_model_params={},
            module_parameters_to_patch=[type(network.layer)],
            mapping_routine=naive_map,
            transistor=True,
            programming_routine=None,
            scheme=memtorch.bh.Scheme.SingleColumn,
            tile_shape=tile_shape,
            max_input_voltage=1.0,
            ADC_resolution=ADC_resolution,
            quant_method=quant_method,
            use_bindings=use_bindings,
        )
        patched_network.tune_()
        patched_network.disable_legacy()
