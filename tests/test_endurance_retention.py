import pytest
import numpy as np
import copy
import math
import torch
import memtorch
from memtorch.bh.nonideality.NonIdeality import apply_nonidealities
from memtorch.mn.Module import supported_module_parameters
from memtorch.bh.nonideality.endurance_retention_models.empirical_metal_oxide_RRAM import (
    model_endurance_retention,
)
from memtorch.bh.nonideality.endurance_retention_models.conductance_drift import (
    model_conductance_drift,
)


@pytest.mark.parametrize("tile_shape", [None, (128, 128), (10, 20)])
@pytest.mark.parametrize(
    "operation_mode",
    [
        memtorch.bh.nonideality.endurance_retention_models.OperationMode.sudden,
        memtorch.bh.nonideality.endurance_retention_models.OperationMode.gradual,
    ],
)
@pytest.mark.parametrize("temperature", [350, None])
def test_model_endurance_retention_retention(
    debug_patched_networks,
    tile_shape,
    operation_mode,
    temperature,
    time=1e4,
    p_lrs=[1, 0, 0, 0],
    stable_resistance_lrs=100,
    p_hrs=[1, 0, 0, 0],
    stable_resistance_hrs=1000,
    cell_size=None,
):
    device = torch.device("cpu" if "cpu" in memtorch.__version__ else "cuda")
    patched_networks = debug_patched_networks(tile_shape, None)
    for patched_network in patched_networks:
        patched_network = apply_nonidealities(
            copy.deepcopy(patched_network),
            non_idealities=[memtorch.bh.nonideality.NonIdeality.Retention],
            time=float(time),
            retention_model=memtorch.bh.nonideality.endurance_retention_models.model_endurance_retention,
            retention_model_kwargs={
                "operation_mode": operation_mode,
                "p_lrs": p_lrs,
                "stable_resistance_lrs": stable_resistance_lrs,
                "p_hrs": p_hrs,
                "stable_resistance_hrs": stable_resistance_hrs,
                "cell_size": cell_size,
                "temperature": temperature,
            },
        )


@pytest.mark.parametrize("tile_shape", [None, (128, 128), (10, 20)])
@pytest.mark.parametrize(
    "operation_mode",
    [
        memtorch.bh.nonideality.endurance_retention_models.OperationMode.sudden,
        memtorch.bh.nonideality.endurance_retention_models.OperationMode.gradual,
    ],
)
@pytest.mark.parametrize("temperature", [350, None])
def test_model_endurance_retention_endurance(
    debug_patched_networks,
    tile_shape,
    operation_mode,
    temperature,
    x=1e4,
    p_lrs=[1, 0, 0, 0],
    stable_resistance_lrs=100,
    p_hrs=[1, 0, 0, 0],
    stable_resistance_hrs=1000,
    cell_size=None,
):
    device = torch.device("cpu" if "cpu" in memtorch.__version__ else "cuda")
    patched_networks = debug_patched_networks(tile_shape, None)
    for patched_network in patched_networks:
        patched_network = apply_nonidealities(
            copy.deepcopy(patched_network),
            non_idealities=[memtorch.bh.nonideality.NonIdeality.Endurance],
            x=float(x),
            endurance_model=memtorch.bh.nonideality.endurance_retention_models.model_endurance_retention,
            endurance_model_kwargs={
                "operation_mode": operation_mode,
                "p_lrs": p_lrs,
                "stable_resistance_lrs": stable_resistance_lrs,
                "p_hrs": p_hrs,
                "stable_resistance_hrs": stable_resistance_hrs,
                "cell_size": cell_size,
                "temperature": temperature,
            },
        )
