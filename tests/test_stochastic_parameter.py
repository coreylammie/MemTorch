import copy
import math
import random

import numpy as np
import pytest
import torch

import memtorch
from memtorch.bh.crossbar.Program import naive_program
from memtorch.map.Parameter import naive_map
from memtorch.mn.Module import patch_model


@pytest.mark.parametrize("mean, std", [(0, 2.5), (0, 5)])
def test_stochastic_parameter(mean, std):
    with pytest.raises(Exception):
        stochastic_parameter = memtorch.bh.StochasticParameter(invalid_arg=None)

    stochastic_parameter = memtorch.bh.StochasticParameter(loc=mean, scale=std)
    assert type(stochastic_parameter(return_mean=True).item()) == float

    class TestObject:
        def __init__(self, test_parameter):
            args = memtorch.bh.unpack_parameters(locals())
            self.test_parameter = args.test_parameter

    kwargs = {"test_parameter": stochastic_parameter}
    test_object = TestObject(**kwargs)
    assert isinstance(
        test_object.test_parameter, (int, float, complex)
    ) and not isinstance(test_object.test_parameter, bool)
    parameter = random.random()
    kwargs = {"test_parameter": parameter}
    test_object = TestObject(**kwargs)
    assert isinstance(
        test_object.test_parameter, (int, float, complex)
    ) and not isinstance(test_object.test_parameter, bool)
    assert (
        type(memtorch.bh.StochasticParameter(loc=mean, scale=std, function=False))
        == float
    )


def test_resample_r_off_r_on(debug_networks):
    networks = debug_networks
    for network in networks:
        with pytest.raises(Exception):
            patched_network = patch_model(
                copy.deepcopy(network),
                memristor_model=memtorch.bh.memristor.LinearIonDrift,
                memristor_model_params={
                    "r_off": memtorch.bh.StochasticParameter(loc=1, scale=0),
                    "r_on": memtorch.bh.StochasticParameter(loc=1, scale=0),
                },
                module_parameters_to_patch=[type(network.layer)],
                mapping_routine=naive_map,
                transistor=True,
                programming_routine=None,
                scheme=memtorch.bh.Scheme.SingleColumn,
            )
        with pytest.raises(Exception):
            patched_network = patch_model(
                copy.deepcopy(network),
                memristor_model=memtorch.bh.memristor.LinearIonDrift,
                memristor_model_params={"r_off": 1, "r_on": 1},
                module_parameters_to_patch=[type(network.layer)],
                mapping_routine=naive_map,
                transistor=True,
                programming_routine=None,
                scheme=memtorch.bh.Scheme.SingleColumn,
            )
