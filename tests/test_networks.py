import pytest
import numpy as np
import copy
import math
import torch
import memtorch
from memtorch.mn.Module import patch_model
from memtorch.map.Parameter import naive_map
from memtorch.bh.crossbar.Program import naive_program


@pytest.mark.parametrize('tile_shape', [None, (128, 128), (10, 20)])
def test_networks(debug_networks, tile_shape):
    networks = debug_networks
    for network in networks:
        patched_network = patch_model(copy.deepcopy(network),
                                      memristor_model=memtorch.bh.memristor.LinearIonDrift,
                                      memristor_model_params={},
                                      module_parameters_to_patch=[type(network.layer)],
                                      mapping_routine=naive_map,
                                      transistor=True,
                                      programming_routine=None,
                                      scheme=memtorch.bh.Scheme.SingleColumn,
                                      tile_shape=tile_shape)
        patched_network.tune_()
