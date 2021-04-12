"""
Conductance drift model as described in:
I. Boybat et al., "Impact of conductance drift on multi-PCM synaptic architectures", 2018 Non-Volatile Memory Technology Symposium (NVMTS), 2018.
"""
import numpy as np
import torch


def model_conductance_drift(layer, x, initial_time=1e-12, drift_coefficient=0.1):
    """
    Parameters
    ----------
    layer : memtorch.mn
        A memrstive layer.
    x : float
        Retention time (s). Denoted using x for sake of consistency with other models.
    initial_time : float
        Initial time that corresponds with initial conductance values.
    drift_coefficient : float
        Drift coefficient. For PCM devices, drift_coefficient typically has a value of 0.1 for the amorphous phase.

    Returns
    -------
    memtorch.mn
        The patched memristive layer.
    """
    time = x
    if initial_time == 0.0:
        initial_time = 1e-12

    assert (
        time >= 0 and initial_time >= 0 and time > initial_time
    ), "time and/or initial_time are/is invalid."
    assert (
        drift_coefficient >= 0 and drift_coefficient <= 1
    ), "drift_coefficient must be >=0 and <= 1."
    for i in range(len(layer.crossbars)):
        initial_conductance = layer.crossbars[i].conductance_matrix
        initial_conductance = initial_conductance * (
            (time / initial_time) ** (-drift_coefficient)
        )
        layer.crossbars[i].conductance_matrix = initial_conductance

    return layer
