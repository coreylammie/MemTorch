import pytest
import numpy as np
import copy
import math
import torch
import memtorch
from memtorch.mn.Module import patch_model
from memtorch.map.Parameter import naive_map
from memtorch.bh.crossbar.Program import naive_program


def test_networks(debug_networks):
    networks = debug_networks
    for network in networks:
        patched_network = patch_model(copy.deepcopy(network),
                                      memristor_model=memtorch.bh.memristor.LinearIonDrift,
                                      memristor_model_params={},
                                      module_parameters_to_patch=[type(network.layer)],
                                      mapping_routine=naive_map,
                                      transistor=True,
                                      programming_routine=None,
                                      scheme=memtorch.bh.Scheme.SingleColumn)
        patched_network.tune_()
