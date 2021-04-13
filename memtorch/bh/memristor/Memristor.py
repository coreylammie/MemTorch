import math
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import signal

import memtorch


class Memristor(ABC):
    """
    Parameters
    ----------
    r_off : float
        Off (maximum) resistance of the device (ohms).
    r_on : float
        On (minimum) resistance of the device (ohms).
    time_series_resolution : float
        Time series resolution (s).
    pos_write_threshold : float
        Positive write threshold voltage (V).
    neg_write_threshold : float
        Negative write threshold voltage (V).
    """

    def __init__(
        self,
        r_off,
        r_on,
        time_series_resolution,
        pos_write_threshold=0,
        neg_write_threshold=0,
    ):
        self.r_off = r_off
        self.r_on = r_on
        self.time_series_resolution = time_series_resolution
        self.pos_write_threshold = pos_write_threshold
        self.neg_write_threshold = neg_write_threshold
        self.g = None
        self.finite_states = None

    @abstractmethod
    def simulate(self, voltage_signal):
        """Method to determine the equivalent conductance of a memristive device when a given voltage signal is applied.

        Parameters
        ----------
        voltage_signal : torch.Tensor
            A discrete voltage signal with resolution time_series_resolution.

        Returns
        -------
        torch.Tensor
            A tensor containing the equivalent device conductance for each simulated timestep.
        """
        return

    @abstractmethod
    def set_conductance(self, conductance):
        """Method to manually set the conductance of a memristive device.

        Parameters
        ----------
        conductance : float
                Conductance to set.
        """
        return

    def get_resistance(self):
        """
        Method to determine the resistance of a memristive device.

        Returns
        -------
        float
            The devices resistance (ohms).
        """
        return 1 / self.g

    @abstractmethod
    def plot_hysteresis_loop(
        self,
        memristor,
        duration,
        voltage_signal_amplitude,
        voltage_signal_frequency,
        log_scale=False,
        return_result=False,
    ):
        """Method to plot the hysteresis loop of a given device.

        Parameters
        ----------
        memristor : memtorch.bh.memristor.Memristor.Memristor
            Memristor.
        duration : float
            Duration (s).
        voltage_signal_amplitude: float
            Voltage signal amplitude (V).
        voltage_signal_frequency : float
            Voltage signal frequency (Hz)
        log_scale : bool
            Plot the y-axis (current) using a symmetrical log scale (True).
        return_result: bool
            Voltage and current signals are returned (True).
        """
        return plot_hysteresis_loop(
            memristor,
            duration,
            voltage_signal_amplitude,
            voltage_signal_frequency,
            log_scale,
            return_result,
        )

    @abstractmethod
    def plot_bipolar_switching_behaviour(
        self,
        memristor,
        voltage_signal_amplitude,
        voltage_signal_frequency,
        log_scale=True,
        return_result=False,
    ):
        """Method to plot the DC bipolar switching behaviour of a given device.

        Parameters
        ----------
        memristor : memtorch.bh.memristor.Memristor.Memristor
            Memristor.
        voltage_signal_amplitude: float
            Voltage signal amplitude (V).
        voltage_signal_frequency : float
            Voltage signal frequency (Hz)
        log_scale : bool
            Plot the y-axis (current) using a symmetrical log scale (True).
        return_result: bool
            Voltage and current signals are returned (True).
        """
        return plot_bipolar_switching_behaviour(
            memristor,
            voltage_signal_amplitude,
            voltage_signal_frequency,
            log_scale,
            return_result,
        )


def plot_hysteresis_loop(
    memristor_model,
    duration,
    voltage_signal_amplitude,
    voltage_signal_frequency,
    log_scale=False,
    return_result=False,
):
    """Method to plot the hysteresis loop of a given device.

    Parameters
    ----------
    memristor_model : memtorch.bh.memristor.Memristor.Memristor
        Memristor model.
    duration : float
        Duration (s).
    voltage_signal_amplitude: float
        Voltage signal amplitude (V).
    voltage_signal_frequency : float
        Voltage signal frequency (Hz)
    log_scale : bool
        Plot the y-axis (current) using a symmetrical log scale (True).
    return_result: bool
        Voltage and current signals are returned (True).

    Returns
    -------
    tuple
        Voltage and current signals.
    """
    time_signal = np.arange(
        0,
        duration + memristor_model.time_series_resolution,
        step=memristor_model.time_series_resolution,
    )
    voltage_signal = voltage_signal_amplitude * np.sin(
        2 * math.pi * voltage_signal_frequency * time_signal
    )
    current_signal = memristor_model.simulate(voltage_signal, return_current=True)
    if return_result:
        return voltage_signal, current_signal
    else:
        plt.figure()
        plt.title("Hysteresis Loop")
        plt.xlabel("Voltage (V)")
        plt.plot(voltage_signal, current_signal)
        if log_scale:
            plt.ylabel("log10(Current (A))")
            plt.yscale("symlog")
        else:
            plt.ylabel("Current (A)")

        plt.show()
        return


def plot_bipolar_switching_behaviour(
    memristor_model,
    voltage_signal_amplitude,
    voltage_signal_frequency,
    log_scale=True,
    return_result=False,
):
    """Method to plot the DC bipolar switching behaviour of a given device.

    Parameters
    ----------
    memristor_model : memtorch.bh.memristor.Memristor.Memristor
        Memristor model.
    voltage_signal_amplitude: float
        Voltage signal amplitude (V).
    voltage_signal_frequency : float
        Voltage signal frequency (Hz)
    log_scale : bool
        Plot the y-axis (current) using a symmetrical log scale (True).
    return_result: bool
        Voltage and current signals are returned (True).
    """

    def gen_triangle_waveform(n_points, amplitude):
        def triangle_iterator_generator(n, amp):
            y = 0
            x = 0
            s = amplitude / (n / 4)
            while x < n_points:
                yield y
                y += s
                if abs(y) > amplitude:
                    s *= -1

                x += 1

        return np.fromiter(triangle_iterator_generator(n_points, amplitude), "d")

    time_signal = np.arange(
        0,
        (1 / voltage_signal_frequency) + memristor_model.time_series_resolution,
        step=memristor_model.time_series_resolution,
    )
    voltage_signal = gen_triangle_waveform(len(time_signal), voltage_signal_amplitude)
    current_signal = memristor_model.simulate(voltage_signal, return_current=True)
    if return_result:
        return voltage_signal, current_signal
    else:
        plt.figure()
        plt.title("Bipolar Switching Behaviour (DC)")
        plt.xlabel("Voltage (V)")
        if log_scale:
            plt.ylabel("|log10(Current (A))|")
            plt.yscale("log")
        else:
            plt.ylabel("|Current (A)|")

        plt.plot(voltage_signal, abs(current_signal))
        plt.show()
        return
