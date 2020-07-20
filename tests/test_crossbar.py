import pytest
import numpy as np
import copy
import math
import random
import torch
import memtorch
from memtorch.map.Parameter import naive_map
from memtorch.bh.crossbar.Crossbar import simulate_matmul
from memtorch.bh.crossbar.Program import naive_program, gen_programming_signal

@pytest.mark.parametrize('shape', [(10, 10), (5, 10)])
def test_crossbar(shape):
    memristor_model = memtorch.bh.memristor.LinearIonDrift
    memristor_model_params = {'time_series_resolution': 1e-3}
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
    assert torch.all(torch.isclose(simulate_matmul(inputs, crossbar.devices).float(),
                     torch.matmul(inputs, conductance_matrix.T).float(), rtol=1e-2))
    assert torch.all(torch.isclose(simulate_matmul(inputs, crossbar.devices, parallelize=True).float(),
                     torch.matmul(inputs, conductance_matrix.T).float(), rtol=1e-2))
    programming_signal = gen_programming_signal(1, 1e-3, 1e-3, 1, memristor_model_params['time_series_resolution'])
    assert type(programming_signal) == tuple
    with pytest.raises(AssertionError):
        gen_programming_signal(1, 1e-4, 1e-4, 1, memristor_model_params['time_series_resolution'])

    point = (0, 0)
    row, column = point
    conductance_to_program = 1 / ((crossbar.devices[row][column].r_on + crossbar.devices[row][column].r_off / 2))
    crossbar.devices[row][column] = naive_program(crossbar, (row, column), conductance_to_program, pulse_duration=1e-2)
    assert math.isclose(conductance_to_program, crossbar.devices[row][column].g, abs_tol=1e-5)
    with pytest.raises(AssertionError):
        naive_program(crossbar, (row, column), -1)
