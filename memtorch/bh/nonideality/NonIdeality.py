import memtorch
import memtorch.mn
from memtorch.bh.nonideality.FiniteConductanceStates import apply_finite_conductance_states
from memtorch.bh.nonideality.DeviceFaults import apply_device_faults
from memtorch.bh.nonideality.NonLinear import apply_non_linear
from memtorch.mn.Module import supported_module_parameters
import numpy as np
import torch
import math
from enum import Enum, auto, unique


@unique
class NonIdeality(Enum):
    """NonIdeality enumeration."""
    FiniteConductanceStates = auto()
    DeviceFaults = auto()
    NonLinear = auto()


def apply_nonidealities(model, non_idealities, **kwargs):
    """Method to apply non-idealities to a torch.nn.Module instance with memristive layers.

    Parameters
    ----------
    model : torch.nn.Module
        torch.nn.Module instance.
    nonidealities : memtorch.bh.nonideality.NonIdeality.NonIdeality, tuple
        Non-linearitites to model.

    Returns
    -------
    torch.nn.Module
        Patched instance.
    """
    for i, (name, m) in enumerate(list(model.named_modules())):
        if type(m) in supported_module_parameters.values():
            if 'cpu' not in memtorch.__version__ and len(name.split('.')) > 1:
                name = name.split('.')[1]

            for non_ideality in non_idealities:
                if non_ideality == NonIdeality.FiniteConductanceStates:
                    required(kwargs, ['conductance_states'], 'memtorch.bh.nonideality.NonIdeality.FiniteConductanceStates')
                    if hasattr(model, 'module'):
                        setattr(model.module, name, apply_finite_conductance_states(m, kwargs['conductance_states']))
                    else:
                        setattr(model, name, apply_finite_conductance_states(m, kwargs['conductance_states']))

                if non_ideality == NonIdeality.DeviceFaults:
                    required(kwargs, ['lrs_proportion', 'hrs_proportion', 'electroform_proportion'], 'memtorch.bh.nonideality.NonIdeality.DeviceFaults')
                    if hasattr(model, 'module'):
                        setattr(model.module, name, apply_device_faults(m, kwargs['lrs_proportion'], kwargs['hrs_proportion'], kwargs['electroform_proportion']))
                    else:
                        setattr(model, name, apply_device_faults(m, kwargs['lrs_proportion'], kwargs['hrs_proportion'], kwargs['electroform_proportion']))

                if non_ideality == NonIdeality.NonLinear:
                    if 'simulate' in kwargs:
                        if kwargs['simulate'] == True:
                            if hasattr(model, 'module'):
                                setattr(model.module, name, apply_non_linear(m, simulate=True))
                            else:
                                setattr(model, name, apply_non_linear(m, simulate=True))
                        else:
                            required(kwargs, ['sweep_duration', 'sweep_voltage_signal_amplitude', 'sweep_voltage_signal_frequency'], 'memtorch.bh.nonideality.NonIdeality.NonLinear')
                            if hasattr(model, 'module'):
                                setattr(model.module, name, apply_non_linear(m, kwargs['sweep_duration'], kwargs['sweep_voltage_signal_amplitude'], kwargs['sweep_voltage_signal_frequency']))
                            else:
                                setattr(model, name, apply_non_linear(m, kwargs['sweep_duration'], kwargs['sweep_voltage_signal_amplitude'], kwargs['sweep_voltage_signal_frequency']))
                    else:
                        required(kwargs, ['sweep_duration', 'sweep_voltage_signal_amplitude', 'sweep_voltage_signal_frequency'], 'memtorch.bh.nonideality.NonIdeality.NonLinear')
                        if hasattr(model, 'module'):
                            setattr(model.module, name, apply_non_linear(m, kwargs['sweep_duration'], kwargs['sweep_voltage_signal_amplitude'], kwargs['sweep_voltage_signal_frequency']))
                        else:
                            setattr(model, name, apply_non_linear(m, kwargs['sweep_duration'], kwargs['sweep_voltage_signal_amplitude'], kwargs['sweep_voltage_signal_frequency']))

    return model

def required(kwargs, arguments, call):
    """Method to check is required arguments in **kwargs are present.

    Parameters
    ----------
    kwargs : **kwargs
        Keyword-arguments.
    arguments : list of str
        Arguments which are required to be present.
    call : str
        Function to call.
    """
    for argument in arguments:
        assert kwargs[argument] is not None, '%s is required when calling %s' % (argument, call)
