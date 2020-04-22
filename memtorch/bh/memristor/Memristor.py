from abc import ABC, abstractmethod
import torch
import memtorch
import numpy as np
import math
import matplotlib.pyplot as plt


class Memristor(ABC):
    """
    Parameters
    ----------
    time_series_resolution : float
        Time series resolution (s).
    pos_write_threshold : float
        Positive write threshold voltage (V).
    neg_write_threshold : float
        Negative write threshold voltage (V).
    """

    def __init__(self, time_series_resolution, pos_write_threshold=0, neg_write_threshold=0):
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

    def get_resistance(self):
        """
        Method to determine the resistance of a memristive device.

        Returns
        -------
        float
            The devices resistance (ohms).
        """
        return 1 / self.g

    def plot_hysteresis_loop(self, memristor, duration, voltage_signal_amplitude, voltage_signal_frequency, return_result=False):
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
        return_result: bool
            Voltage and current signals are returned (True).
        """
        return plot_hysteresis_loop(memristor, duration, voltage_signal_amplitude, voltage_signal_frequency, return_result)


def plot_hysteresis_loop(memristor_model, duration, voltage_signal_amplitude, voltage_signal_frequency, return_result=False):
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
    return_result: bool
        Voltage and current signals are returned (True).

    Returns
    -------
    tuple
        Voltage and current signals.
    """
    time_signal = np.arange(0, duration + memristor_model.time_series_resolution, step=memristor_model.time_series_resolution)
    voltage_signal = voltage_signal_amplitude * np.sin(2 * math.pi * voltage_signal_frequency * time_signal)
    current_signal = memristor_model.simulate(voltage_signal, return_current=True)
    if return_result:
        return voltage_signal, current_signal
    else:
        plt.figure()
        plt.title('Hysteresis Loop')
        plt.xlabel('Voltage (V)')
        plt.ylabel('Current (A)')
        plt.plot(voltage_signal, current_signal)
        plt.show()
        return
