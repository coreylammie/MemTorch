import pytest
import numpy as np
import copy
import math
import random
import torch
import memtorch

@pytest.mark.parametrize('mean, std', [(0, 0), (0, 5)])
def test_stochastic_parameter(mean, std):
    stochastic_parameter = memtorch.bh.StochasticParameter(mean, std=std)
    class TestObject():
        def __init__(self, test_parameter):
            args = memtorch.bh.unpack_parameters(locals())
            self.test_parameter = args.test_parameter

    kwargs = {'test_parameter': stochastic_parameter}
    test_object = TestObject(**kwargs)
    assert isinstance(test_object.test_parameter, (int, float, complex)) and not isinstance(test_object.test_parameter, bool)
    parameter = random.random()
    kwargs = {'test_parameter': parameter}
    test_object = TestObject(**kwargs)
    assert isinstance(test_object.test_parameter, (int, float, complex)) and not isinstance(test_object.test_parameter, bool)
