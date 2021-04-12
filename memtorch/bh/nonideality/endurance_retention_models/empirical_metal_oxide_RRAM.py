"""
Empirical Metal-Oxide RRAM Device Endurance and Retention Model as described in:
C. Lammie, M. Rahimi Azghadi and D. Ielmini, "Empirical Metal-Oxide RRAM Device Endurance and Retention Model For Deep Learning Simulations", Semiconductor Science and Technology, 2021.
"""
import enum
import importlib
import math
from enum import Enum, auto

import numpy as np
import scipy
import torch
from scipy.interpolate import interp1d


class OperationMode(Enum):
    sudden = auto()
    gradual = auto()


def scale_p_0(p_0, p_1, v_stop, v_stop_min, v_stop_max, v_stop_optimal, cell_size=10):
    """Method to scale p_0 to introduce dependence on v_stop

    Parameters
    ----------
    p_0 : float
        Unscaled p_0 parameter.
    p_1 : float
        p_1 parameter.
    v_stop : float
        v_stop value to determine p_0 for.
    v_stop_min : float
        minimum v_stop value.
    v_stop_max : float
        maximum v_stop value.
    v_stop_optimal : float
        optimal v_stop value (endurance).
    cell_size : float
        Device cell size (nm).

    Returns
    -------
    float
        Scaled p_0 value.
    """
    assert (
        v_stop_max > v_stop_min
        and v_stop_optimal >= v_stop_min
        and v_stop_optimal <= v_stop_max
    ), "v_stop_max must be larger than v_stop_min and v_stop_optimal must be >= v_stop_min and <= v_stop_max."
    if cell_size is None:
        cell_size = 10

    scale_input = interp1d([v_stop_min, v_stop_max], [0, 1])
    scaled_input = scale_input(v_stop)
    y = p_0 * np.exp(p_1 * cell_size)
    k = np.log10(y) / (1 - (2 * scale_input(v_stop_optimal) - 1) ** (2))
    return (10 ** (k * (1 - (2 * scaled_input - 1) ** (2)))) / (np.exp(p_1 * cell_size))


def model_endurance_retention_gradual(
    initial_resistance,
    x,
    p_0,
    p_1,
    p_2,
    p_3,
    threshold,
    temperature_constant,
    cell_size=None,
):
    """Method to model gradual endurance_retention failure.

    Parameters
    ----------
    initial_resistance : tensor
        Initial resistance values.
     x : float
        Energy (J) / SET-RESET cycles / Retention time (s).
    p_0 : float
        p_0 fitting parameter.
    p_1 : float
        p_1 fitting parameter.
    p_2 : float
        p_2 fitting parameter.
    p_3 : float
        p_3 fitting parameter.
    threshold : float
        Threshold for x.
    temperature_threshold : float
        Temperature threshold (K) in which the device begins to fail.
    cell_size : float
        Device cell size (nm).

    Returns
    -------
    tensor
        Updated resistance values.
    """
    return 10 ** (
        p_3 * (p_1 * cell_size + p_2 * temperature_constant) * np.log10(x)
        + np.log10(initial_resistance)
        - p_3 * (p_1 * cell_size + p_2 * temperature_constant) * np.log10(threshold)
    )


def model_endurance_retention(
    layer,
    operation_mode,
    x,
    p_lrs,
    stable_resistance_lrs,
    p_hrs,
    stable_resistance_hrs,
    cell_size,
    temperature,
    temperature_threshold=298,
):
    """Method to model endurance and retention characteristics.

    Parameters
    ----------
    layer : memtorch.mn
        A memrstive layer.
    operation_mode: memtorch.bh.endurance_retention_models.OperationMode
        Failure operational mode (sudden or gradual).
    x : float
        x parameter.
    p_lrs : list
        Low resistance state p_0, p_1, p_2, and p_3 values.
    stable_resistance_lrs : float
        Stable low resistance state.
    p_hrs : list
        High resistance state p_0, p_1, p_2, and p_3 values.
    stable_resistance_hrs : float
        Stable high resistance state.
    cell_size : float
        Device cell size (nm).
    temperature : float
        Operational temperature (K).
    temperature_threshold : float
        Temperature threshold (K) in which the device begins to fail.

    Returns
    -------
    memtorch.mn
        The patched memristive layer.
    """
    assert (len(p_lrs) == 4 or p_lrs is None) and (
        len(p_hrs) == 4 or p_hrs is None
    ), "p_lrs or p_hrs are of invalid length."
    if cell_size is None:
        cell_size = 10

    if temperature is None or temperature_threshold is None:
        temperature_constant = 0
    else:
        temperature_constant = min(temperature_threshold / temperature, 1)

    if p_lrs is not None:
        if temperature is None:
            threshold_lrs = p_lrs[0] * np.exp(p_lrs[1])
        else:
            threshold_lrs = p_lrs[0] * np.exp(
                p_lrs[1] * cell_size + p_lrs[2] * temperature_constant
            )

    if p_hrs is not None:
        if temperature is None:
            threshold_hrs = p_hrs[0] * np.exp(p_hrs[1])
        else:
            threshold_hrs = p_hrs[0] * np.exp(
                p_hrs[1] * cell_size + p_hrs[2] * temperature_constant
            )

    for i in range(len(layer.crossbars)):
        initial_resistance = 1 / layer.crossbars[i].conductance_matrix
        convergence_point = (
            layer.crossbars[i].r_on_mean + layer.crossbars[i].r_off_mean / 2
        )
        if x > threshold_lrs:
            if (
                operation_mode == OperationMode.gradual
                and p_lrs is not None
                and initial_resistance[
                    initial_resistance < convergence_point
                ].nelement()
                > 0
            ):
                initial_resistance[
                    initial_resistance < convergence_point
                ] = model_endurance_retention_gradual(
                    initial_resistance=initial_resistance[
                        initial_resistance < convergence_point
                    ],
                    x=x,
                    p_0=p_lrs[0],
                    p_1=p_lrs[1],
                    p_2=p_lrs[2],
                    p_3=p_lrs[3],
                    threshold=threshold_lrs,
                    temperature_constant=temperature_constant,
                    cell_size=cell_size,
                )
            elif operation_mode == OperationMode.sudden:
                initial_resistance[
                    initial_resistance < convergence_point
                ] = stable_resistance_lrs

        if x > threshold_hrs:
            if (
                operation_mode == OperationMode.gradual
                and p_hrs is not None
                and initial_resistance[
                    initial_resistance > convergence_point
                ].nelement()
                > 0
            ):
                initial_resistance[
                    initial_resistance > convergence_point
                ] = model_endurance_retention_gradual(
                    initial_resistance=initial_resistance[
                        initial_resistance > convergence_point
                    ],
                    x=x,
                    p_0=p_hrs[0],
                    p_1=p_hrs[1],
                    p_2=p_hrs[2],
                    p_3=p_hrs[3],
                    threshold=threshold_hrs,
                    temperature_constant=temperature_constant,
                    cell_size=cell_size,
                )
            elif operation_mode == OperationMode.sudden and x > threshold_hrs:
                initial_resistance[
                    initial_resistance > convergence_point
                ] = stable_resistance_hrs

        layer.crossbars[i].conductance_matrix = 1 / initial_resistance

    return layer
