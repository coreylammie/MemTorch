import memtorch
from .Memristor import Memristor as Memristor
from memtorch.utils import convert_range
import numpy as np
import torch
import math


class VTEAM(Memristor):
    """VTEAM memristor model.

    Parameters
    ----------
    time_series_resolution : float
        Time series resolution (s).
    r_off : float
        Off (maximum) resistance of the device (ohms).
    r_on : float
        On (minimum) resistance of the device (ohms).
    d : float
        Device length (m).
    k_on : float
        k_on model parameter.
    k_off: float
        k_off model parameter.
    alpha_on : float
        alpha_on model parameter.
    alpha_off : float
        alpha_off model parameter.
    v_on : float
        Positive write threshold voltage (V).
    v_off : float
        Negative write threshold voltage (V).
    x_on : float
        x_on model parameter.
    x_off : float
        x_off model parameter.
    """
    def __init__(self,
                 time_series_resolution=1e-10,
                 r_off=1000,
                 r_on=50,
                 d=3e-9,
                 k_on=-10,
                 k_off=5e-4,
                 alpha_on=3,
                 alpha_off=1,
                 v_on=-0.2,
                 v_off=0.02,
                 x_on=0,
                 x_off=3e-9,
                 **kwargs):

        args = memtorch.bh.unpack_parameters(locals())
        super(VTEAM, self).__init__(args.time_series_resolution, args.v_off, args.v_on)
        self.r_off = args.r_off
        self.r_on = args.r_on
        self.d = args.d
        self.k_on = args.k_on
        self.k_off = args.k_off
        self.alpha_on = args.alpha_on
        self.alpha_off = args.alpha_off
        self.v_on = args.v_on
        self.v_off = args.v_off
        self.x_on = args.x_on
        self.x_off = args.x_off
        self.g = 1 / self.r_on
        self.x = self.x_on
        self.lamda = np.log(self.r_off / self.r_on)

    def dxdt(self, voltage):
        """ Method to determine the derivative of the state variable.

        Parameters
        ----------
        voltage : float
            The current applied voltage (V).

        Returns
        -------
        float
            The derivative of the state variable.
        """
        if voltage >= self.v_off:
            return self.k_off * (((voltage / self.v_off) - 1) ** self.alpha_off)
        elif voltage <= self.v_on:
            return self.k_on * (((voltage / self.v_on) - 1) ** self.alpha_on)
        else:
            return 0

    def current(self, voltage):
        """Method to determine the current of the model given an applied voltage.

        Parameters
        ----------
        voltage : float
            The current applied voltage (V).

        Returns
        -------
        float
            The observed current (A).
        """
        return voltage / (self.r_off * self.x / self.d + self.r_on * (1 - self.x / self.d))

    def simulate(self, voltage_signal, return_current=False):
        if return_current:
            current = np.zeros(len(voltage_signal))

        np.seterr(all='raise')
        for t in range(0, len(voltage_signal)):
            self.x = self.x + (self.dxdt(voltage_signal[t]) * self.time_series_resolution)
            if self.x >= self.d or self.x <= 0:
                if self.x >= self.d:
                    self.x = self.d
                else:
                    self.x = 0

            current_ = self.current(voltage_signal[t])
            if voltage_signal[t] != 0:
                self.g = current_ / voltage_signal[t]

            if return_current:
                current[t] = current_

        if return_current:
            return current

    def set_conductance(self, conductance):
        assert (1 / conductance) >= self.r_on and conductance <= self.r_off, 'Conductance to program must be between g_off and g_on.'
        self.x = self.d * ((1 / conductance) - self.r_on) / (self.r_off - self.r_on)
        self.g = conductance

    def plot_hysteresis_loop(self, duration=200e-9, voltage_signal_amplitude=1, voltage_signal_frequency=50e6, return_result=False):
        return super(VTEAM, self).plot_hysteresis_loop(self, duration=duration, voltage_signal_amplitude=voltage_signal_amplitude, voltage_signal_frequency=voltage_signal_frequency, return_result=return_result)
