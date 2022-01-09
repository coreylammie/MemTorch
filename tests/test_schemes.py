import copy

import pytest
import torch

import memtorch
from memtorch.bh.crossbar.Program import naive_program
from memtorch.map.Parameter import naive_map
from memtorch.mn.Module import patch_model


@pytest.mark.parametrize("tile_shape", [None, (128, 128), (10, 20)])
def test_schemes(debug_networks, tile_shape):
    networks = debug_networks
    for scheme in memtorch.bh.Scheme:
        for network in networks:
            patched_network = patch_model(
                copy.deepcopy(network),
                memristor_model=memtorch.bh.memristor.LinearIonDrift,
                memristor_model_params={},
                module_parameters_to_patch=[type(network.layer)],
                mapping_routine=naive_map,
                transistor=True,
                programming_routine=None,
                scheme=scheme,
                tile_shape=tile_shape,
            )
            assert patched_network.layer.crossbars is not None
