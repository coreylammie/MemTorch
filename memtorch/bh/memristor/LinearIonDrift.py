import math

import numpy as np
import torch

import memtorch
from memtorch.utils import clip, convert_range

from .Memristor import Memristor as Memristor


class LinearIonDrift(Memristor):
    """Linear Ion behvaioural drift model.

    Parameters
    ----------
    time_series_resolution : float
        Time series resolution (s).
    u_v : float
        Dopant drift mobility of the device material.
    d : float
        Device length (m).
    r_on : float
        On (minimum) resistance of the device (ohms).
    r_off : float
        Off (maximum) resistance of the device (ohms).
    pos_write_threshold: float
        Positive write threshold voltage (V).
    neg_write_threshold : float
        Negative write threshold voltage (V).
    p : int
        Joglekar window p constant.
    """

    def __init__(
        self,
        time_series_resolution=1e-4,
        u_v=1e-14,
        d=10e-9,
        r_on=100,
        r_off=16e3,
        pos_write_threshold=0.55,
        neg_write_threshold=-0.55,
        p=1,
        **kwargs
    ):

        args = memtorch.bh.unpack_parameters(locals())
        super(LinearIonDrift, self).__init__(
            args.r_off,
            args.r_on,
            args.time_series_resolution,
            args.pos_write_threshold,
            args.neg_write_threshold,
        )
        self.u_v = args.u_v
        self.d = args.d
        self.r_i = args.r_on
        self.p = args.p
        self.g = 1 / self.r_i
        self.x = convert_range(self.r_i, self.r_on, self.r_off, 0, 1)

    def simulate(self, voltage_signal, return_current=False):
        len_voltage_signal = 1
        try:
            len_voltage_signal = len(voltage_signal)
        except:
            voltage_signal = [voltage_signal]

        if return_current:
            current = np.zeros(len_voltage_signal)

        np.seterr(all="raise")
        for t in range(0, len_voltage_signal):
            current_ = self.current(voltage_signal[t])
            if (
                voltage_signal[t] >= self.pos_write_threshold
                or voltage_signal[t] <= self.neg_write_threshold
            ):
                self.x = self.x + self.dxdt(current_) * self.time_series_resolution
                self.x = max(min(1.0, self.x), 0.0)

            try:
                self.g = 1 / ((self.r_on * self.x) + (self.r_off * (1 - self.x)))
            except:
                self.g = 0

            if self.g > (1 / self.r_on):
                self.g = 1 / self.r_on
            elif self.g < (1 / self.r_off):
                self.g = 1 / self.r_off

            if return_current:
                current[t] = current_

        if return_current:
            return current

    def set_conductance(self, conductance):
        conductance = clip(conductance, 1 / self.r_off, 1 / self.r_on)
        self.x = convert_range(1 / conductance, self.r_on, self.r_off, 0, 1)
        self.g = conductance

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
        return voltage / (((self.r_on * self.x) + (self.r_off * (1 - self.x))))

    def dxdt(self, current):
        """Method to determine the derivative of the state variable, dx/dt.

        Parameters
        ----------
        current : float
            The observed current (A).

        Returns
        -------
        float
            The derivative of the state variable, dx/dt.
        """
        return (
            self.u_v
            * (self.r_on / (self.d ** 2))
            * current
            * memtorch.bh.memristor.window.Jogelkar(self.x, self.p)
        )

    def plot_hysteresis_loop(
        self,
        duration=4,
        voltage_signal_amplitude=5,
        voltage_signal_frequency=2.5,
        return_result=False,
    ):
        return super().plot_hysteresis_loop(
            self,
            duration=duration,
            voltage_signal_amplitude=voltage_signal_amplitude,
            voltage_signal_frequency=voltage_signal_frequency,
            return_result=return_result,
        )

    def plot_bipolar_switching_behaviour(
        self,
        voltage_signal_amplitude=5,
        voltage_signal_frequency=2.5,
        log_scale=True,
        return_result=False,
    ):
        return super().plot_bipolar_switching_behaviour(
            self,
            voltage_signal_amplitude=voltage_signal_amplitude,
            voltage_signal_frequency=voltage_signal_frequency,
            log_scale=log_scale,
            return_result=return_result,
        )
