import math

import numpy as np

import memtorch
from memtorch.utils import clip
from .Memristor import Memristor as Memristor


class Data_Driven2021(Memristor):
    """An updated Data-Driven Verilog-A ReRAM Model. Based on the model used in the article at the following address:
    https://arxiv.org/abs/2012.02267. The main difference with the 2018 version (simply Data_Driven.py) is that it is
    less computationally demanding and achieves similar results. The creator of the data driven model as decided to
    return to the 2017 version for this reason in his recent works.
    The default parameters were determined experimentally.

    Parameters
    ----------
    time_series_resolution : float
        Time series resolution (s).
    r_off : float
        Off (maximum) resistance of the device (ohms).
    r_on : float
        On (minimum) resistance of the device (ohms).
    A_p : float
        A_p model parameter.
    A_n : float
        A_n model parameter.
    t_p : float
        t_p model parameter.
    t_n : float
        t_n model parameter.
    k_p : float
        k_p model parameter.
    k_n : float
        k_n model parameter.
    r_p : float
        r_p voltage-dependent resistive boundary function coefficients.
    r_n : float
        r_n voltage-dependent resistive boundary function coefficients.
    a_p : float
        a_p model parameter.
    a_n : float
        a_n model parameter.
    b_p : float
        b_p model parameter.
    b_n : float
        b_n model parameter.
    """

    def __init__(
        self,
        time_series_resolution=1e-10,
        r_off=3000,
        r_on=1600,
        A_p=600.10075,
        A_n=-34.5988399,
        t_p=-0.0212028,
        t_n=-0.05343997,
        k_p=5.11e-4,
        k_n=1.17e-3,
        r_p=[2699.2336, -672.930205],  # a0_p, a1_p
        r_n=[649.413746, -1474.32358],  # a0_n, a1_n
        a_p=0.32046175,
        a_n=0.32046175,
        b_p=2.71689828,
        b_n=2.71689828,
        **kwargs
    ):

        args = memtorch.bh.unpack_parameters(locals())
        super(Data_Driven2021, self).__init__(
            args.r_off, args.r_on, args.time_series_resolution, 0, 0
        )
        self.A_p = args.A_p
        self.A_n = args.A_n
        self.t_p = args.t_p
        self.t_n = args.t_n
        self.k_p = args.k_p
        self.k_n = args.k_n
        self.r_p = args.r_p
        self.r_n = args.r_n
        self.a_p = args.a_p
        self.a_n = args.a_n
        self.b_p = args.b_p
        self.b_n = args.b_n
        self.g = 1 / args.r_on

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
            self.g = 1 / self.resistance(voltage_signal[t])
            if return_current:
                current[t] = current_
        if return_current:
            return current

    def r_pn(self, voltage, a0, a1):
        """Function to return rp(v) or rn(v)
        From the 2017/2021 paper model calculations

        Parameters
        ----------
        voltage : float
            The current applied voltage (V).
        a0, a1: float
            The value of a0 and a1

        Returns
        -------
        float
        The rp or rn resistance (Ω).
        """
        return a0 + a1 * voltage

    def s_pn(self, voltage, A, t):
        """Function to return sp(v) or sn(v)
        From the 2017/2021 paper model calculations

        Parameters
        ----------
        voltage : float
        The current applied voltage (V).
        A, t: float
        The value of model params A (A_p or A_n) or t(t_p or t_n)

        Returns
        -------
        float
        The sp or sn variability.
        """
        return A * (math.exp(abs(voltage) / t) - 1)

    def dRdt(self, voltage):
        """Function to return dR/dT
        From the 2017/2021 paper model calculations

        Parameters
        ----------
        voltage : float
        The current applied voltage (V).
        a0, a1: float
        The value of a0 or a1

        Returns
        -------
        float
        The derivative with respect to time of the resistance
        """

        R = 1 / self.g
        if voltage > 0:
            r_p = self.r_pn(voltage, self.r_p[0], self.r_p[1])
            s_p = self.s_pn(voltage, self.A_p, self.t_p)
            return s_p * (r_p - R) ** 2
        if voltage <= 0:
            r_n = self.r_pn(voltage, self.r_n[0], self.r_n[1])
            s_n = self.s_pn(voltage, self.A_n, self.t_n)
            return s_n * (R - r_n) ** 2

        return

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
        if voltage > 0:
            return self.a_p * self.g * math.sinh(self.b_p * voltage)
        else:
            return self.a_n * self.g * math.sinh(self.b_n * voltage)

    def resistance(self, voltage):
        """Method to determine the resistance of the model given an applied voltage.
        Using the 2017/2021 model

        Parameters
        ----------
        voltage : float
            The current applied voltage (V).

        Returns
        -------
        float
            The observed resistance (Ω).
        """
        R0 = 1 / self.g
        if voltage > 0:
            r_p = self.r_pn(voltage, self.r_p[0], self.r_p[1])
            s_p = self.s_pn(voltage, self.A_p, self.t_p)
            resistance_ = (
                R0 + (s_p * r_p * (r_p - R0)) * self.time_series_resolution
            ) / (1 + s_p * (r_p - R0) * self.time_series_resolution)
            if resistance_ < r_p:
                return R0
            else:
                return max(
                    min(resistance_, self.r_off), self.r_on
                )  # Artificially confine the resistance between r_on and r_off

        elif voltage < 0:
            r_n = self.r_pn(voltage, self.r_n[0], self.r_n[1])
            s_n = self.s_pn(voltage, self.A_n, self.t_n)
            resistance_ = (
                R0 + (s_n * r_n * (r_n - R0)) * self.time_series_resolution
            ) / (1 + s_n * (r_n - R0) * self.time_series_resolution)
            if resistance_ > r_n:
                return R0
            else:
                return max(
                    min(resistance_, self.r_off), self.r_on
                )  # Artificially confine the resistance between r_on and r_off

        else:
            return 1 / self.g

    def set_conductance(self, conductance):
        conductance = clip(conductance, 1 / self.r_off, 1 / self.r_on)
        self.g = conductance

    def plot_hysteresis_loop(
        self,
        duration=1e-3,
        voltage_signal_amplitude=1.5,
        voltage_signal_frequency=10e3,
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
        voltage_signal_amplitude=1.5,
        voltage_signal_frequency=10e3,
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
