import math

import numpy as np
import torch

import memtorch
from memtorch.utils import clip, convert_range

from .Memristor import Memristor as Memristor


class Stanford_PKU(Memristor):
    """Stanford PKU memristor model (https://nano.stanford.edu/stanford-rram-model).

    Parameters
    ----------
    time_series_resolution : float
        Time series resolution (s).
    r_off : float
        Off (maximum) resistance of the device (ohms).
    r_on : float
        On (minimum) resistance of the device (ohms).
    gap_init : float
        Initial gap distance (m).
    g_0 : float
        g_0 model parameter.
    V_0 : float
        V_0 model parameter.
    I_0 : float
        I_0 model parameter.
    read_voltage : float
        Read voltage (V) to determine the device's conductance.
    T_init : float
        Initial room tempurature.
    R_th : float
        Thermal resistance.
    gamma_init : float
        gamma_init model parameter.
    beta : float
        beta model parameter.
    t_ox : float
        Oxide thickness (m).
    F_min : float
        Minimum field requirement to enhance gap formation.
    vel_0 : float
        vel_0 model parameter.
    E_a : float
        Activation energy.
    a_0 : float
        Atom spacing.
    delta_g_init : float
        Initial delta_g value.
    model_switch : int
        Switch to select standard model (0) or dynamic model (1).
    T_crit : float
        Threshold temperature (K) for significant random variations.
    T_smth : float
        Activation energy for vacancy generation.
    """

    def __init__(
        self,
        time_series_resolution=1e-4,
        r_off=218586,
        r_on=542,
        gap_init=2e-10,
        g_0=0.25e-9,
        V_0=0.25,
        I_0=1000e-6,
        read_voltage=0.1,
        T_init=298,
        R_th=2.1e3,
        gamma_init=16,
        beta=0.8,
        t_ox=12e-9,
        F_min=1.4e9,
        vel_0=10,
        E_a=0.6,
        a_0=0.25e-9,
        delta_g_init=0.02,
        model_switch=0,
        T_crit=450,
        T_smth=500,
        **kwargs
    ):

        args = memtorch.bh.unpack_parameters(locals())
        super(Stanford_PKU, self).__init__(
            args.r_off, args.r_on, args.time_series_resolution, 0, 0
        )
        self.gap_init = args.gap_init
        self.g_0 = args.g_0
        self.V_0 = args.V_0
        self.I_0 = args.I_0
        self.read_voltage = args.read_voltage
        self.T_init = args.T_init
        self.R_th = args.R_th
        self.gamma_init = args.gamma_init
        self.beta = args.beta
        self.t_ox = args.t_ox
        self.F_min = args.F_min
        self.vel_0 = args.vel_0
        self.E_a = args.E_a
        self.a_0 = args.a_0
        self.delta_g_init = args.delta_g_init
        self.model_switch = args.model_switch
        self.T_crit = args.T_crit
        self.T_smth = args.T_smth
        gap_min = self.g_0 * np.log(
            self.I_0
            * np.sinh(self.read_voltage / self.V_0)
            / (self.read_voltage / self.r_on)
        )
        gap_max = self.g_0 * np.log(
            self.I_0
            * np.sinh(self.read_voltage / self.V_0)
            / (self.read_voltage / self.r_off)
        )
        assert (
            gap_min > 0 and gap_max > 0 and gap_max > gap_min
        ), "Invalid gap length bounds (min: %f, max: %f) encountered." % (
            gap_min,
            gap_max,
        )
        self.gap_min = gap_min
        self.gap_max = gap_max
        self.gap = max(min(gap_init, gap_max), gap_min)
        self.g = self.current(self.read_voltage) / self.read_voltage

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
        return self.I_0 * np.exp(-self.gap / self.g_0) * np.sinh(voltage / self.V_0)

    def T_current(self, voltage, current):
        """Method to determine the thermal current of the model given an applied voltage and current.

        Parameters
        ----------
        voltage : float
            The current applied voltage (V).
        current : float
            The current applied current (A).

        Returns
        -------
        float
            The observed thermal current (A).
        """
        return self.T_init + abs(voltage * current * self.R_th)

    def dg_dt(self, voltage, current):
        """Method to determine the derivative of the gap length.

        Parameters
        ----------
        voltage : float
            The current applied voltage (V).
        current : float
            The current applied current (A).

        Returns
        -------
        float
            The derivative of the gap length.
        """
        q = 1.6e-19
        k_b = 1.3806503e-23
        gamma = self.gamma_init - self.beta * np.power(self.gap / 1e-9, 3)
        if gamma * abs(voltage) / self.t_ox < self.F_min:
            gamma = 0

        delta_g = self.delta_g_init * self.model_switch
        T_current_eval = self.T_current(voltage, current)
        return -self.vel_0 * np.exp(-q * self.E_a / k_b / T_current_eval) * np.sinh(
            gamma * self.a_0 / self.t_ox * q * voltage / k_b / T_current_eval
        ) + np.random.normal(loc=0, scale=1) * delta_g / (
            1 + np.exp((self.T_crit - T_current_eval) / self.T_smth)
        )

    def simulate(self, voltage_signal, return_current=False):
        len_voltage_signal = 1
        try:
            len_voltage_signal = len(voltage_signal)
        except:
            voltage_signal = [voltage_signal]

        if return_current:
            current = np.zeros(len_voltage_signal)

        np.seterr(all="raise")
        current_ = 0
        for t in range(0, len_voltage_signal):
            self.gap = (
                self.gap
                + self.dg_dt(voltage_signal[t], current_) * self.time_series_resolution
            )
            self.gap = max(min(self.gap, self.gap_max), self.gap_min)
            if voltage_signal[t] != 0:
                self.g = current_ / voltage_signal[t]

            current_ = self.current(voltage_signal[t])
            if return_current:
                current[t] = current_

        if return_current:
            return current

    def set_conductance(self, conductance):
        conductance = clip(conductance, 1 / self.r_off, 1 / self.r_on)
        gap = self.g_0 * np.log(
            self.I_0
            * np.sinh(self.read_voltage / self.V_0)
            / (self.read_voltage / (1 / conductance))
        )
        self.gap = max(min(gap, self.gap_max), self.gap_min)

    def plot_hysteresis_loop(
        self,
        duration=0.5,
        voltage_signal_amplitude=1.5,
        voltage_signal_frequency=10,
        log_scale=False,
        return_result=False,
    ):
        return super().plot_hysteresis_loop(
            self,
            duration=duration,
            voltage_signal_amplitude=voltage_signal_amplitude,
            voltage_signal_frequency=voltage_signal_frequency,
            log_scale=log_scale,
            return_result=return_result,
        )

    def plot_bipolar_switching_behaviour(
        self,
        voltage_signal_amplitude=1.5,
        voltage_signal_frequency=0.05,
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
