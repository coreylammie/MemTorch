import matplotlib
import numpy as np
import pytest
import torch

import memtorch


def get_subclasses(cls):
    for subclass in cls.__subclasses__():
        yield from get_subclasses(subclass)
        yield subclass


matplotlib.use("Agg")
memristor_models = list(get_subclasses(memtorch.bh.memristor.Memristor))


@pytest.mark.parametrize("model", memristor_models)
def test_model(model):
    model_instance = model()
    assert model_instance.g is not None
    assert model_instance.r_on is not None
    assert model_instance.r_off is not None
    assert model_instance.time_series_resolution is not None


@pytest.mark.parametrize("model", memristor_models)
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_plot_hysteresis_loop(model):
    model_instance = model()
    voltage_signal, current_signal = model_instance.plot_hysteresis_loop(
        return_result=True
    )
    assert voltage_signal is not None and current_signal is not None
    assert len(voltage_signal) == len(current_signal)
    assert model_instance.plot_hysteresis_loop(return_result=False) is None
    assert model_instance.plot_bipolar_switching_behaviour(return_result=False) is None


@pytest.mark.parametrize("model", memristor_models)
def test_simulate(model):
    model_instance = model()
    voltage_signal = np.zeros((100, 1))
    assert len(model_instance.simulate(voltage_signal, return_current=True)) == 100
    assert model_instance.simulate(voltage_signal, return_current=False) is None
