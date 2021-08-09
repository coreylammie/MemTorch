import copy
import math

import matplotlib.pyplot as plt
import numpy as np
import torch

import memtorch


def apply_non_linear(
    layer,
    sweep_duration=1,
    sweep_voltage_signal_amplitude=1,
    sweep_voltage_signal_frequency=1,
    num_conductance_states=None,
    simulate=False,
):
    """Method to model non_linear I/V characteristics for devices within a memristive layer.

    Parameters
    ----------
    layer : memtorch.mn
        A memrstive layer.
    sweep_duration : float
        Voltage sweep duration (s).
    sweep_voltage_signal_amplitude : float
        Voltage sweep amplitude (V).
    sweep_voltage_signal_frequency : float
        Voltage sweep frequency (Hz).
    num_conductance_states : int, optional
        Number of finite conductance states to model. None indicates finite states are not to be modeled.
    simulate : bool, optional
        Each device is simulated during inference (True).

    Returns
    -------
    memtorch.mn
        The patched memristive layer.
    """

    def apply_non_linear_to_device(
        device,
        sweep_duration,
        sweep_voltage_signal_amplitude,
        sweep_voltage_signal_frequency,
    ):
        time_signal = np.arange(
            0,
            sweep_duration + device.time_series_resolution,
            step=device.time_series_resolution,
        )
        voltage_signal = np.cos(
            2 * math.pi * sweep_voltage_signal_frequency * time_signal
        )
        current_signal = copy.deepcopy(device).simulate(
            voltage_signal, return_current=True
        )

        def det_current(voltage):
            if np.isnan(voltage.cpu()):
                return 0

            assert (
                abs(voltage) <= sweep_voltage_signal_amplitude
            ), "voltage must be between -sweep_voltage_signal_amplitude and sweep_voltage_signal_amplitude."
            if voltage < 0:
                return (
                    -1
                    * current_signal[::-1][
                        np.searchsorted(
                            voltage_signal[::-1], -1 * voltage.cpu(), side="left"
                        )
                    ]
                )
            else:
                return current_signal[::-1][
                    np.searchsorted(voltage_signal[::-1], voltage.cpu(), side="left")
                ]

        device.det_current = det_current
        return device

    def apply_non_linear_to_crossbar(
        crossbar,
        sweep_duration,
        sweep_voltage_signal_amplitude,
        sweep_voltage_signal_frequency,
    ):
        assert (
            len(crossbar.devices.shape) == 2 or len(crossbar.devices.shape) == 3
        ), "Invalid devices shape."
        if len(crossbar.devices.shape) == 2:
            for row in range(0, crossbar.rows):
                for column in range(0, crossbar.columns):
                    crossbar.devices[row, column] = apply_non_linear_to_device(
                        crossbar.devices[row, column],
                        sweep_duration,
                        sweep_voltage_signal_amplitude,
                        sweep_voltage_signal_frequency,
                    )
        else:
            for i in range(0, crossbar.devices.shape[0]):
                for j in range(crossbar.devices.shape[1]):
                    for k in range(crossbar.devices.shape[2]):
                        crossbar.devices[i, j, k] = apply_non_linear_to_device(
                            crossbar.devices[i, j, k],
                            sweep_duration,
                            sweep_voltage_signal_amplitude,
                            sweep_voltage_signal_frequency,
                        )

        return crossbar

    layer.non_linear = True
    if simulate:
        layer.simulate = True
    else:
        if num_conductance_states is None:
            for i in range(len(layer.crossbars)):
                layer.crossbars[i] = apply_non_linear_to_crossbar(
                    layer.crossbars[i],
                    sweep_duration,
                    sweep_voltage_signal_amplitude,
                    sweep_voltage_signal_frequency,
                )
        else:
            raise ("To be implemented.")

    return layer
