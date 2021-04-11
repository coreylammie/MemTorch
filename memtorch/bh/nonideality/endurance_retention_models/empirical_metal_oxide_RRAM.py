"""
Wrapper for the SST-Reproducibility GeneralModel class from the empirical model described in:
C. Lammie, M. Rahimi Azghadi and D. Ielmini, "Empirical metal-oxide RRAM device endurance and retention model for deep learning simulations", Semiconductor Science and Technology, 2021.
"""
from scipy.interpolate import interp1d
import scipy
import numpy as np
import torch
import importlib

endurance_retention_model = importlib.import_module(
    ".GeneralModel", "memtorch.submodules.SST-Reproducibility"
)


OperationMode = endurance_retention_model.OperationMode  # OperationMode enum


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
    scale_input = interp1d([v_stop_min, v_stop_max], [0, 1])
    scaled_input = scale_input(v_stop)
    y = p_0 * np.exp(p_1 * cell_size)
    k = np.log10(y) / (1 - (2 * scale_input(v_stop_optimal) - 1) ** (2))
    return (10 ** (k * (1 - (2 * scaled_input - 1) ** (2)))) / (np.exp(p_1 * cell_size))


def model_endurance_retention(
    layer,
    operation_mode,
    x,
    p_lrs,
    stable_resistance_lrs,
    p_hrs,
    stable_resistance_hrs,
    cell_size,
    tempurature,
    tempurature_threshold=298,
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
        Low resistance state p_0, p_1, and p_2 values.
    stable_resistance_lrs : float
        Stable low resistance state.
    p_hrs : list
        High resistance state p_0, p_1, and p_2 values.
    stable_resistance_hrs : float
        Stable high resistance state.
    cell_size : float
        Device cell size (nm).
    tempurature : float
        Operational tempurature (K).
    tempurature_threshold : float
        Tempurature threshold (K) in which the device begins to fail.

    Returns
    -------
    memtorch.mn
        The patched memristive layer.

    """

    assert (len(p_lrs) == 3 or p_lrs is None) and (
        len(p_hrs) == 3 or p_hrs is None
    ), "p_lrs or p_hrs are of invalid length."
    if cell_size is None:
        cell_size = 10

    if p_lrs is not None:
        if tempurature is None:
            threshold_lrs = p_lrs[0] * np.exp(p_lrs[1])
        else:
            threshold_lrs = p_lrs[0] * np.exp(
                p_lrs[1] * cell_size
                + p_lrs[2] * np.min(tempurature_threshold / tempurature, 1)
            )

    if p_hrs is not None:
        if tempurature is None:
            threshold_hrs = p_hrs[0] * np.exp(p_hrs[1])
        else:
            threshold_hrs = p_hrs[0] * np.exp(
                p_hrs[1] * cell_size
                + p_hrs[2] * np.min(tempurature_threshold / tempurature, 1)
            )

    for i in range(len(layer.crossbars)):
        initial_resistance = 1 / layer.crossbars[i].conductance_matrix
        convergence_point = (
            layer.crossbars[i].r_on_mean + layer.crossbars[i].r_off_mean / 2
        )
        general_model = endurance_retention_model.GeneralModel(operation_mode)
        if x > threshold_lrs:
            if (
                general_model.operation_mode == OperationMode.gradual
                and p_lrs is not None
                and initial_resistance[
                    initial_resistance < convergence_point
                ].nelement()
                > 0
            ):
                initial_resistance[
                    initial_resistance < convergence_point
                ] = general_model.model(
                    x,
                    initial_resistance[initial_resistance < convergence_point],
                    p_lrs[0],
                    p_lrs[1],
                    p_lrs[2],
                    p_lrs[3],
                    stable_resistance_lrs,
                    cell_size,
                    tempurature,
                    tempurature_threshold,
                )
            elif general_model.operation_mode == OperationMode.sudden:
                initial_resistance[
                    initial_resistance < convergence_point
                ] = stable_resistance_lrs

        if x > threshold_hrs:
            if (
                general_model.operation_mode == OperationMode.gradual
                and p_hrs is not None
                and initial_resistance[
                    initial_resistance > convergence_point
                ].nelement()
                > 0
            ):
                initial_resistance[
                    initial_resistance > convergence_point
                ] = general_model.model(
                    x,
                    initial_resistance[initial_resistance > convergence_point],
                    p_hrs[0],
                    p_hrs[1],
                    p_hrs[2],
                    p_hrs[3],
                    stable_resistance_hrs,
                    cell_size,
                    tempurature,
                    tempurature_threshold,
                )
            elif (
                general_model.operation_mode == OperationMode.sudden
                and x > threshold_hrs
            ):
                initial_resistance[
                    initial_resistance > convergence_point
                ] = stable_resistance_hrs

        layer.crossbars[i].conductance_matrix = 1 / initial_resistance

    return layer
