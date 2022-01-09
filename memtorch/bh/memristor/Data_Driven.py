import math

import numpy as np

import memtorch
from memtorch.utils import clip
from .Memristor import Memristor as Memristor


class Data_Driven(Memristor):
    """A Data-Driven Verilog-A ReRAM Model (https://eprints.soton.ac.uk/411693/).

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
    eta : int
        Switching direction to stimulus polarity.
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
        time_series_resolution=1e-8,
        r_off=17e3,
        r_on=1280,
        A_p=743.47,
        A_n=-68012.28374,
        t_p=6.51,
        t_n=0.31645,
        k_p=5.11e-4,
        k_n=1.17e-3,
        r_p=[16719, 0],
        r_n=[29304.82557, 23692.77225],
        eta=1,
        a_p=0.24,
        a_n=0.24,
        b_p=3,
        b_n=3,
        **kwargs
    ):

        args = memtorch.bh.unpack_parameters(locals())
        super(Data_Driven, self).__init__(
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
        self.eta = args.eta
        self.a_p = args.a_p
        self.a_n = args.a_n
        self.b_p = args.b_p
        self.b_n = args.b_n
        self.g = 1 / self.r_on

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

        Parameters
        ----------
        voltage : float
            The current applied voltage (V).

        Returns
        -------
        float
            The observed resistance (Ω).
        """

        def r_pn(voltage, r_pn):
            sum = 0
            for m_pn in range(0, len(r_pn)):
                sum += r_pn[m_pn] * (voltage ** (m_pn))

            return sum

        if voltage > 0:
            r_pn_eval = r_pn(voltage, self.r_p)
            resistance_ = (
                np.log(
                    np.exp(self.eta * self.k_p * r_pn_eval)
                    + np.exp(
                        -self.eta
                        * self.k_p
                        * (self.A_p * (math.exp(self.t_p * abs(voltage)) - 1))
                        * self.time_series_resolution
                    )
                    * (
                        np.exp(self.eta * self.k_p * (1 / self.g))
                        - np.exp(self.eta * self.k_p * r_pn_eval)
                    )
                )
                / self.k_p
            )
            if resistance_ > self.eta * r_pn_eval:
                return 1 / self.g
            else:
                return max(
                    min(resistance_, self.r_off), self.r_on
                )  # Artificially confine the resistance between r_on and r_off
        elif voltage < 0:
            r_pn_eval = r_pn(voltage, self.r_n)
            resistance_ = (
                -np.log(
                    np.exp(
                        -self.eta * self.k_n * (1 / self.g)
                        + self.eta
                        * self.k_n
                        * (self.A_n * (-1 + np.exp(self.t_n * abs(voltage))))
                        * self.time_series_resolution
                    )
                    - np.exp(-self.eta * self.k_n * r_pn_eval)
                    * (
                        -1
                        + np.exp(
                            self.eta
                            * self.k_n
                            * (self.A_n * (-1 + np.exp(self.t_n * abs(voltage))))
                            * self.time_series_resolution
                        )
                    )
                )
                / self.k_n
            )
            if resistance_ < self.eta * r_pn_eval:
                return 1 / self.g
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
