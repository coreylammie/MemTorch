import copy
import math

import numpy as np
import pytest
import torch

import memtorch
from memtorch.bh.nonideality.DeviceFaults import apply_cycle_variability
from memtorch.bh.nonideality.endurance_retention_models.conductance_drift import (
    model_conductance_drift,
)
from memtorch.bh.nonideality.endurance_retention_models.empirical_metal_oxide_RRAM import (
    model_endurance_retention,
)
from memtorch.bh.nonideality.NonIdeality import apply_nonidealities
from memtorch.mn.Module import supported_module_parameters


@pytest.mark.parametrize("tile_shape", [None, (128, 128), (10, 20)])
@pytest.mark.parametrize("quant_method", memtorch.bh.Quantize.quant_methods + [None])
def test_device_faults(debug_patched_networks, tile_shape, quant_method):
    device = torch.device("cpu" if "cpu" in memtorch.__version__ else "cuda")
    patched_networks = debug_patched_networks(tile_shape, quant_method)
    for patched_network in patched_networks:
        patched_network_lrs = apply_nonidealities(
            copy.deepcopy(patched_network),
            non_idealities=[memtorch.bh.nonideality.NonIdeality.DeviceFaults],
            lrs_proportion=0.5,
            hrs_proportion=0,
            electroform_proportion=0,
        )
        patched_tensor_lrs = (
            patched_network_lrs.layer.crossbars[0].conductance_matrix.float().to(device)
        )
        lrs = (
            torch.tensor(
                1
                / np.vectorize(lambda x: x.r_on)(
                    patched_network_lrs.layer.crossbars[0].devices
                )
            )
            .float()
            .to(device)
        )
        lrs_percentage = (
            sum(torch.isclose(patched_tensor_lrs, lrs).view(-1)).item()
            / patched_tensor_lrs.numel()
        )
        patched_network_hrs = apply_nonidealities(
            copy.deepcopy(patched_network),
            non_idealities=[memtorch.bh.nonideality.NonIdeality.DeviceFaults],
            lrs_proportion=0,
            hrs_proportion=0.25,
            electroform_proportion=0.25,
        )
        patched_tensor_hrs = (
            patched_network_hrs.layer.crossbars[0].conductance_matrix.float().to(device)
        )
        hrs = (
            torch.tensor(
                1
                / np.vectorize(lambda x: x.r_off)(
                    patched_network_hrs.layer.crossbars[0].devices
                )
            )
            .float()
            .to(device)
        )
        hrs_percentage = (
            sum(torch.isclose(patched_tensor_hrs, hrs).view(-1)).item()
            / patched_tensor_hrs.numel()
        )
        assert (
            lrs_percentage >= 0.25 and hrs_percentage >= 0.25
        )  # To account for some degree of stochasticity


@pytest.mark.parametrize("tile_shape", [None, (128, 128), (10, 20)])
@pytest.mark.parametrize("quant_method", memtorch.bh.Quantize.quant_methods + [None])
def test_finite_conductance_states(
    debug_patched_networks, tile_shape, quant_method, conductance_states=5
):
    device = torch.device("cpu" if "cpu" in memtorch.__version__ else "cuda")
    patched_networks = debug_patched_networks(tile_shape, quant_method)
    for patched_network in patched_networks:
        patched_network_finite_states = apply_nonidealities(
            copy.deepcopy(patched_network),
            non_idealities=[
                memtorch.bh.nonideality.NonIdeality.FiniteConductanceStates
            ],
            conductance_states=5,
        )
        conductance_matrix = patched_network.layer.crossbars[0].conductance_matrix
        quantized_conductance_matrix = patched_network_finite_states.layer.crossbars[
            0
        ].conductance_matrix
        quantized_conductance_matrix_unique = quantized_conductance_matrix.unique()
        valid_values = torch.linspace(
            patched_network.layer.crossbars[0].conductance_matrix.min(),
            patched_network.layer.crossbars[0].conductance_matrix.max(),
            conductance_states,
        ).float()
        assert any(
            [
                bool(val)
                for val in [
                    torch.isclose(
                        quantized_conductance_matrix_unique, valid_value
                    ).any()
                    for valid_value in valid_values
                ]
            ]
        )
        assert conductance_matrix.shape == quantized_conductance_matrix.shape


@pytest.mark.parametrize("tile_shape", [None, (128, 128), (10, 20)])
@pytest.mark.parametrize("parallelize", [True, False])
@pytest.mark.parametrize("quant_method", memtorch.bh.Quantize.quant_methods + [None])
def test_cycle_variability(
    debug_patched_networks, tile_shape, parallelize, quant_method, std=10
):
    patched_networks = debug_patched_networks(tile_shape, quant_method)
    for patched_network in patched_networks:
        for i, (name, m) in enumerate(list(patched_network.named_modules())):
            if type(m) in supported_module_parameters.values():
                if "cpu" not in memtorch.__version__ and len(name.split(".")) > 1:
                    name = name.split(".")[1]

                if hasattr(patched_network, "module"):
                    with pytest.raises(Exception):
                        setattr(
                            patched_network.module,
                            name,
                            apply_cycle_variability(
                                m,
                                parallelize=parallelize,
                                r_off_kwargs={"invalid_arg": None},
                                r_on_kwargs={"invalid_arg": None},
                            ),
                        )

                    setattr(
                        patched_network.module,
                        name,
                        apply_cycle_variability(
                            m,
                            parallelize=parallelize,
                            r_off_kwargs={
                                "loc": m.crossbars[0].r_off_mean,
                                "scale": std * 2,
                            },
                            r_on_kwargs={"loc": m.crossbars[0].r_on_mean, "scale": std},
                        ),
                    )
                else:
                    with pytest.raises(Exception):
                        setattr(
                            patched_network,
                            name,
                            apply_cycle_variability(
                                m,
                                parallelize=parallelize,
                                r_off_kwargs={"invalid_arg": None},
                                r_on_kwargs={"invalid_arg": None},
                            ),
                        )

                    setattr(
                        patched_network,
                        name,
                        apply_cycle_variability(
                            m,
                            parallelize=parallelize,
                            r_off_kwargs={
                                "loc": m.crossbars[0].r_off_mean,
                                "scale": std * 2,
                            },
                            r_on_kwargs={"loc": m.crossbars[0].r_on_mean, "scale": std},
                        ),
                    )


@pytest.mark.parametrize("tile_shape", [None, (128, 128), (10, 20)])
@pytest.mark.parametrize("quant_method", memtorch.bh.Quantize.quant_methods + [None])
def test_non_linear(debug_patched_networks, tile_shape, quant_method):
    patched_networks = debug_patched_networks(tile_shape, quant_method)
    for patched_network in patched_networks:
        patched_network_non_linear = apply_nonidealities(
            copy.deepcopy(patched_network),
            non_idealities=[memtorch.bh.nonideality.NonIdeality.NonLinear],
            sweep_duration=2,
            sweep_voltage_signal_amplitude=1,
            sweep_voltage_signal_frequency=0.5,
        )
        patched_network_non_linear.tune_(
            tune_kwargs={
                "<class 'memtorch.mn.Conv1d.Conv1d'>": {
                    "input_batch_size": 1,
                    "input_shape": 2,
                },
                "<class 'memtorch.mn.Conv2d.Conv2d'>": {
                    "input_batch_size": 1,
                    "input_shape": 2,
                },
                "<class 'memtorch.mn.Conv3d.Conv3d'>": {
                    "input_batch_size": 1,
                    "input_shape": 2,
                },
                "<class 'memtorch.mn.Linear.Linear'>": {"input_shape": 2},
            }
        )
        patched_network_non_linear = apply_nonidealities(
            copy.deepcopy(patched_network),
            non_idealities=[memtorch.bh.nonideality.NonIdeality.NonLinear],
            simulate=True,
        )
        patched_network_non_linear.tune_(
            tune_kwargs={
                "<class 'memtorch.mn.Conv1d.Conv1d'>": {
                    "input_batch_size": 1,
                    "input_shape": 2,
                },
                "<class 'memtorch.mn.Conv2d.Conv2d'>": {
                    "input_batch_size": 1,
                    "input_shape": 2,
                },
                "<class 'memtorch.mn.Conv3d.Conv3d'>": {
                    "input_batch_size": 1,
                    "input_shape": 2,
                },
                "<class 'memtorch.mn.Linear.Linear'>": {"input_shape": 2},
            }
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


@pytest.mark.parametrize("v_stop", [0.25, 0.5])
@pytest.mark.parametrize("v_stop_optimal", [0.4, 0.6])
@pytest.mark.parametrize("cell_size", [10, None])
def test_scale_p_0(
    v_stop, v_stop_optimal, cell_size, p_0=1, p_1=1, v_stop_min=0, v_stop_max=1
):
    p_0 = memtorch.bh.nonideality.endurance_retention_models.scale_p_0(
        p_0, p_1, v_stop, v_stop_min, v_stop_max, v_stop_optimal, cell_size
    )


@pytest.mark.parametrize("time", [1e4, 1e6])
@pytest.mark.parametrize("drift_coefficient", [0.1, 0.4])
def test_model_conductance_drift(
    debug_patched_networks,
    time,
    drift_coefficient,
    tile_shape=(128, 128),
    initial_time=1e-12,
):
    device = torch.device("cpu" if "cpu" in memtorch.__version__ else "cuda")
    patched_networks = debug_patched_networks(tile_shape, None)
    for patched_network in patched_networks:
        patched_network = apply_nonidealities(
            copy.deepcopy(patched_network),
            non_idealities=[memtorch.bh.nonideality.NonIdeality.Retention],
            time=float(time),
            retention_model=memtorch.bh.nonideality.endurance_retention_models.model_conductance_drift,
            retention_model_kwargs={
                "initial_time": initial_time,
                "drift_coefficient": drift_coefficient,
            },
        )
