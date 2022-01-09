import copy
import math
import random
import sys

import numpy as np
import pytest
import torch

import memtorch
from memtorch.bh.crossbar.Crossbar import simulate_matmul
from memtorch.bh.crossbar.Program import gen_programming_signal, naive_program
from memtorch.map.Parameter import naive_map


@pytest.mark.filterwarnings("ignore::Warning")
@pytest.mark.parametrize("shape", [(2, 2)])
def test_crossbar(shape):
    device = torch.device("cpu" if "cpu" in memtorch.__version__ else "cuda")
    memristor_model = memtorch.bh.memristor.LinearIonDrift
    memristor_model_params = {"time_series_resolution": 1e-3}
    pos_crossbar = memtorch.bh.crossbar.Crossbar(
        memristor_model, memristor_model_params, shape
    )
    pos_crossbar.devices[0][0].g = 1 / pos_crossbar.r_off_mean
    pos_crossbar.devices[0][1].g = 1 / pos_crossbar.r_on_mean
    neg_crossbar = copy.deepcopy(pos_crossbar)
    pos_conductance_matrix, neg_conductance_matrix = naive_map(
        torch.zeros(shape).uniform_(0, 1),
        memristor_model().r_on,
        memristor_model().r_off,
        memtorch.bh.crossbar.Scheme.DoubleColumn,
    )
    pos_crossbar.write_conductance_matrix(
        pos_conductance_matrix, transistor=False, programming_routine=naive_program
    )
    neg_crossbar.write_conductance_matrix(
        pos_conductance_matrix, transistor=False, programming_routine=naive_program
    )

    crossbar = memtorch.bh.crossbar.Crossbar(
        memristor_model, memristor_model_params, shape
    )
    conductance_matrix = naive_map(
        torch.zeros(shape).uniform_(0, 1),
        memristor_model().r_on,
        memristor_model().r_off,
        memtorch.bh.crossbar.Scheme.SingleColumn,
    )
    crossbar.write_conductance_matrix(conductance_matrix)
    if sys.version_info > (3, 6):
        crossbar.update(from_devices=False, parallelize=True)
        assert torch.all(
            torch.isclose(
                conductance_matrix.T[:, :],
                crossbar.conductance_matrix.cpu()[:, :],
                atol=1e-5,
            )
        )
        assert crossbar.devices[0][0].g == crossbar.conductance_matrix[0][0].item()

    crossbar.update(from_devices=False, parallelize=False)
    assert crossbar.devices[0][0].g == crossbar.conductance_matrix[0][0].item()
    inputs = torch.zeros(shape).uniform_(0, 1)
    assert torch.all(
        torch.isclose(
            simulate_matmul(inputs, crossbar).float(),
            torch.matmul(inputs, conductance_matrix.T).float().to(device),
            rtol=1e-1,
        )
    )
    programming_signal = gen_programming_signal(
        1, 1e-3, 0, 1, memristor_model_params["time_series_resolution"]
    )
    assert type(programming_signal) == tuple
    with pytest.raises(AssertionError):
        gen_programming_signal(
            1, 1e-4, 0, 1, memristor_model_params["time_series_resolution"]
        )

    point = (0, 0)
    row, column = point
    conductance_to_program = random.uniform(
        1 / crossbar.devices[row][column].r_off, 1 / crossbar.devices[row][column].r_on
    )
    crossbar.devices = naive_program(
        crossbar, (row, column), conductance_to_program, rel_tol=0.01
    )
    assert math.isclose(
        conductance_to_program, crossbar.devices[row][column].g, abs_tol=1e-4
    )
    with pytest.raises(AssertionError):
        naive_program(crossbar, (row, column), -1)
