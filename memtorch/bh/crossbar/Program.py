import torch
import memtorch
import numpy as np
import math


def naive_program(crossbar, point, conductance, rel_tol=0.01, pulse_duration=1e-3, refactory_period=0, pos_voltage_level=1.0, neg_voltage_level=-1.0, simulate_neighbours=True):
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

        Returns
        -------
        memtorch.bh.memristor.Memristor.Memristor
            Programmed device.
        """
        row, column = point
        assert (1 / conductance) >= crossbar.devices[row][column].r_on and conductance <= crossbar.devices[row][column].r_off, 'Conductance to program must be between g_off and g_on.'
        if conductance < crossbar.devices[row][column].g:
            time_signal, voltage_signal = gen_programming_signal(1, pulse_duration, refactory_period, pos_voltage_level, crossbar.devices[row][column].time_series_resolution)
            while not math.isclose(conductance, crossbar.devices[row][column].g, rel_tol=rel_tol):
                crossbar.devices[row][column].simulate(voltage_signal)
                if simulate_neighbours:
                    for row_ in range(0, crossbar.rows):
                        if row_ != row:
                            crossbar.devices[row_, column].simulate(voltage_signal / 2)

                    for column_ in range(0, crossbar.columns):
                        if column_ != column:
                            crossbar.devices[row, column_].simulate(voltage_signal / 2)

        elif conductance > crossbar.devices[row][column].g:
            time_signal, voltage_signal = gen_programming_signal(1, pulse_duration, refactory_period, neg_voltage_level, crossbar.devices[row][column].time_series_resolution)
            while not math.isclose(conductance, crossbar.devices[row][column].g, rel_tol=rel_tol):
                crossbar.devices[row][column].simulate(voltage_signal)
                if simulate_neighbours:
                    for row_ in range(0, crossbar.rows):
                        if row_ != row:
                            crossbar.devices[row_, column].simulate(voltage_signal / 2)

                    for column_ in range(0, crossbar.columns):
                        if column_ != column:
                            crossbar.devices[row, column_].simulate(voltage_signal / 2)

        return crossbar.devices

def gen_programming_signal(number_of_pulses, pulse_duration, refactory_period, voltage_level, time_series_resolution):
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
    assert abs(pulse_duration/time_series_resolution - round(pulse_duration/time_series_resolution)) <= assert_tol, 'pulse_duration must be divisible by time_series_resolution.'
    assert abs(refactory_period/time_series_resolution - round(refactory_period/time_series_resolution)) <= assert_tol, 'refactory_period must be divisible by time_series_resolution.'
    time_signal = np.arange(0, duration, step=time_series_resolution)
    period = np.zeros(round(period / time_series_resolution))
    period[0:round(pulse_duration / time_series_resolution)] = voltage_level
    voltage_signal = np.tile(period, number_of_pulses)
    return time_signal, voltage_signal
