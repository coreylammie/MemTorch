import math
from enum import Enum, auto, unique

import numpy as np
import torch

import memtorch
import memtorch.mn
from memtorch.bh.nonideality.DeviceFaults import apply_device_faults
from memtorch.bh.nonideality.Endurance import apply_endurance_model
from memtorch.bh.nonideality.FiniteConductanceStates import (
    apply_finite_conductance_states,
)
from memtorch.bh.nonideality.NonLinear import apply_non_linear
from memtorch.bh.nonideality.Retention import apply_retention_model
from memtorch.mn.Module import supported_module_parameters


@unique
class NonIdeality(Enum):
    """NonIdeality enumeration."""

    FiniteConductanceStates = auto()
    DeviceFaults = auto()
    NonLinear = auto()
    Endurance = auto()
    Retention = auto()


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

    def apply_patched_module(model, patched_module, name, m):
        if name.__contains__("."):
            sequence_container, module = name.split(".")
            if module.isdigit():
                module = int(module)
                model._modules[sequence_container][module] = patched_module
            else:
                setattr(
                    model._modules[sequence_container],
                    "%s" % module,
                    patched_module,
                )
        else:
            model._modules[name] = patched_module

        return model

    for _, (name, m) in enumerate(list(model.named_modules())):
        if type(m) in supported_module_parameters.values():
            for non_ideality in non_idealities:
                if non_ideality == NonIdeality.FiniteConductanceStates:
                    required(
                        kwargs,
                        ["conductance_states"],
                        "memtorch.bh.nonideality.NonIdeality.FiniteConductanceStates",
                    )
                    model = apply_patched_module(
                        model,
                        apply_finite_conductance_states(
                            m, kwargs["conductance_states"]
                        ),
                        name,
                        m,
                    )
                elif non_ideality == NonIdeality.DeviceFaults:
                    required(
                        kwargs,
                        ["lrs_proportion", "hrs_proportion", "electroform_proportion"],
                        "memtorch.bh.nonideality.NonIdeality.DeviceFaults",
                    )
                    model = apply_patched_module(
                        model,
                        apply_device_faults(
                            m,
                            kwargs["lrs_proportion"],
                            kwargs["hrs_proportion"],
                            kwargs["electroform_proportion"],
                        ),
                        name,
                        m,
                    )
                elif non_ideality == NonIdeality.NonLinear:
                    if "simulate" in kwargs:
                        if kwargs["simulate"] == True:
                            model = apply_patched_module(
                                model, apply_non_linear(m, simulate=True), name, m
                            )
                        else:
                            required(
                                kwargs,
                                [
                                    "sweep_duration",
                                    "sweep_voltage_signal_amplitude",
                                    "sweep_voltage_signal_frequency",
                                ],
                                "memtorch.bh.nonideality.NonIdeality.NonLinear",
                            )
                            model = apply_patched_module(
                                model,
                                apply_non_linear(
                                    m,
                                    kwargs["sweep_duration"],
                                    kwargs["sweep_voltage_signal_amplitude"],
                                    kwargs["sweep_voltage_signal_frequency"],
                                ),
                                name,
                                m,
                            )
                    else:
                        required(
                            kwargs,
                            [
                                "sweep_duration",
                                "sweep_voltage_signal_amplitude",
                                "sweep_voltage_signal_frequency",
                            ],
                            "memtorch.bh.nonideality.NonIdeality.NonLinear",
                        )
                        model = apply_patched_module(
                            model,
                            apply_non_linear(
                                m,
                                kwargs["sweep_duration"],
                                kwargs["sweep_voltage_signal_amplitude"],
                                kwargs["sweep_voltage_signal_frequency"],
                            ),
                            name,
                            m,
                        )
                elif non_ideality == NonIdeality.Endurance:
                    required(
                        kwargs,
                        ["x", "endurance_model", "endurance_model_kwargs"],
                        "memtorch.bh.nonideality.Endurance",
                    )
                    model = apply_patched_module(
                        model,
                        apply_endurance_model(
                            layer=m,
                            x=kwargs["x"],
                            endurance_model=kwargs["endurance_model"],
                            **kwargs["endurance_model_kwargs"]
                        ),
                        name,
                        m,
                    )
                elif non_ideality == NonIdeality.Retention:
                    required(
                        kwargs,
                        ["time", "retention_model", "retention_model_kwargs"],
                        "memtorch.bh.nonideality.Retention",
                    )
                    model = apply_patched_module(
                        model,
                        apply_retention_model(
                            layer=m,
                            time=kwargs["time"],
                            retention_model=kwargs["retention_model"],
                            **kwargs["retention_model_kwargs"]
                        ),
                        name,
                        m,
                    )

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
        assert kwargs[argument] is not None, "%s is required when calling %s" % (
            argument,
            call,
        )
