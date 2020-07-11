import pytest
import numpy as np
import copy
import math
import random
import torch
import memtorch
from memtorch.map.Parameter import naive_map
from memtorch.bh.crossbar.Crossbar import simulate_matmul

# @pytest.mark.parametrize('shape', [(10, 10), (20, 20), (10, 20), (20, 10)])
@pytest.mark.parametrize('shape', [(10, 10)])
def test_crossbar(shape):
    memristor_model = memtorch.bh.memristor.LinearIonDrift
    memristor_model_params = {'time_series_resolution': 1e-4}
    crossbar = memtorch.bh.crossbar.Crossbar(memristor_model, memristor_model_params, shape)
    conductance_matrix = naive_map(torch.zeros(shape).uniform_(0, 1),
                                   memristor_model().r_on, memristor_model().r_off,
                                   memtorch.bh.crossbar.Scheme.SingleColumn)
    crossbar.write_conductance_matrix(conductance_matrix)
    crossbar.update(from_devices=False, parallelize=True)
    assert torch.all(conductance_matrix.T[:, :] == crossbar.conductance_matrix.cpu()[:, :])
    assert crossbar.devices[0][0].g == crossbar.conductance_matrix[0][0].item()
    crossbar.update(from_devices=False, parallelize=False)
    assert crossbar.devices[0][0].g == crossbar.conductance_matrix[0][0].item()
    inputs = torch.zeros(shape).uniform_(0, 1)
    assert torch.all(torch.isclose(torch.tensor(simulate_matmul(inputs, crossbar.devices)).float(),
                     torch.matmul(inputs, conductance_matrix.T).float(), rtol=1e-3))
    assert torch.all(torch.isclose(simulate_matmul(inputs, crossbar.devices, parallelize=True),
                     torch.matmul(inputs, conductance_matrix.T), rtol=1e-3))
