import math
import time
import warnings

import numpy as np
import torch

import memtorch


def naive_program(
    crossbar,
    point,
    conductance,
    rel_tol=0.01,
    pulse_duration=1e-3,
    refactory_period=0,
    pos_voltage_level=1.0,
    neg_voltage_level=-1.0,
    timeout=5,
    force_adjustment=1e-3,
    force_adjustment_rel_tol=1e-1,
    force_adjustment_pos_voltage_threshold=0,
    force_adjustment_neg_voltage_threshold=0,
    simulate_neighbours=True,
):
    """Method to program (alter) the conductance of a given device within a crossbar.

    Parameters
    ----------
    crossbar : memtorch.bh.crossbar.Crossbar
        Crossbar containing the device to program.
    point : tuple
        Point to program (row, column).
    conductance : float
        Conductance to program.
    rel_tol : float
        Relative tolerance between the desired conductance and the device's conductance.
    pulse_duration : float
        Duration of the programming pulse (s).
    refactory_period : float
        Duration of the refactory period (s).
    pos_voltage_level : float
        Positive voltage level (V).
    neg_voltage_level : float
        Negative voltage level (V).
    timeout : int
        Timeout (seconds) until stuck devices are unstuck.
    force_adjustment : float
        Adjustment (resistance) to unstick stuck devices.
    force_adjustment_rel_tol : float
        Relative tolerance threshold between a stuck device's conductance and high and low conductance states to force adjust.
    force_adjustment_pos_voltage_threshold : float
        Positive voltage level threshold (V) to enable force adjustment.
    force_adjustment_neg_voltage_threshold : float
        Negative voltage level threshold (V) to enable force adjustment.
    simulate_neighbours : bool
        Simulate neighbours (True).

    Returns
    -------
    memtorch.bh.memristor.Memristor.Memristor
        Programmed device.
    """
    assert (1 / conductance) >= crossbar.devices[
        point
    ].r_on and conductance <= crossbar.devices[
        point
    ].r_off, "Conductance to program must be between g_off and g_on."
    assert (
        len(crossbar.devices.shape) == 2 or len(crossbar.devices.shape) == 3
    ), "Invalid devices shape."
    if len(crossbar.devices.shape) == 3:
        tile, row, column = point
    else:
        row, column = point
        tile = None

    time_signal, pos_voltage_signal = gen_programming_signal(
        1,
        pulse_duration,
        refactory_period,
        pos_voltage_level,
        crossbar.devices[point].time_series_resolution,
    )
    _, neg_voltage_signal = gen_programming_signal(
        1,
        pulse_duration,
        refactory_period,
        neg_voltage_level,
        crossbar.devices[point].time_series_resolution,
    )
    timeout = time.time() + timeout
    iterations = 0
    while not math.isclose(conductance, crossbar.devices[point].g, rel_tol=rel_tol):
        if conductance < crossbar.devices[point].g:
            voltage_signal = neg_voltage_signal
        else:
            voltage_signal = pos_voltage_signal

        previous_g = crossbar.devices[point].g
        crossbar.devices[point].simulate(voltage_signal)
        if simulate_neighbours:
            for row_ in range(0, crossbar.devices.shape[-2]):
                if row_ != row:
                    if tile is not None:
                        idx = (tile, row_, column)
                    else:
                        idx = (row_, column)

                    crossbar.devices[idx].simulate(voltage_signal / 2)

            for column_ in range(0, crossbar.devices.shape[-1]):
                if column_ != column:
                    if tile is not None:
                        idx = (tile, row, column_)
                    else:
                        idx = (row, column_)

                    crossbar.devices[idx].simulate(voltage_signal / 2)

        if crossbar.devices[point].g == previous_g:
            if (
                np.amax(voltage_signal) >= force_adjustment_pos_voltage_threshold
                or np.amin(voltage_signal) <= force_adjustment_neg_voltage_threshold
            ):
                if math.isclose(
                    previous_g,
                    1 / crossbar.devices[point].r_on,
                    rel_tol=force_adjustment_rel_tol,
                ):
                    crossbar.devices[point].set_conductance(
                        crossbar.devices[point].g - force_adjustment
                    )
                elif math.isclose(
                    previous_g,
                    1 / crossbar.devices[point].r_off,
                    rel_tol=force_adjustment_rel_tol,
                ):
                    crossbar.devices[point].set_conductance(
                        crossbar.devices[point].g + force_adjustment
                    )

        iterations += 1
        if iterations % 100 == 0 and time.time() > timeout:
            warnings.warn("Failed to program device to rel_tol (%f)." % rel_tol)
            break

    return crossbar.devices


def gen_programming_signal(
    number_of_pulses,
    pulse_duration,
    refactory_period,
    voltage_level,
    time_series_resolution,
):
    """Method to generate a programming signal using a sequence of pulses.

    Parameters
    ----------
    number_of_pulses : int
        Number of pulses.
    pulse_duration : float
        Duration of the programming pulse (s).
    refactory_period : float
        Duration of the refactory period (s).
    voltage_level : float
        Voltage level (V).
    time_series_resolution : float
        Time series resolution (s).

    Returns
    -------
    tuple
        Tuple containing the generated time and voltage signals.
    """
    period = pulse_duration + refactory_period
    duration = number_of_pulses * period
    assert_tol = 1e-9
    assert (
        abs(
            pulse_duration / time_series_resolution
            - round(pulse_duration / time_series_resolution)
        )
        <= assert_tol
    ), "pulse_duration must be divisible by time_series_resolution."
    assert (
        abs(
            refactory_period / time_series_resolution
            - round(refactory_period / time_series_resolution)
        )
        <= assert_tol
    ), "refactory_period must be divisible by time_series_resolution."
    time_signal = np.arange(0, duration, step=time_series_resolution)
    period = np.zeros(round(period / time_series_resolution))
    period[0 : round(pulse_duration / time_series_resolution)] = voltage_level
    voltage_signal = np.tile(period, number_of_pulses)
    return time_signal, voltage_signal
