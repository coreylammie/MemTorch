import torch
import memtorch
import numpy as np
import math
import copy
from memtorch.bh.nonideality.endurance_retention_models.empirical_metal_oxide_RRAM import (
    model_endurance_retention,
)
from memtorch.bh.nonideality.endurance_retention_models.conductance_drift import (
    model_conductance_drift,
)

valid_retention_models = [model_endurance_retention, model_conductance_drift]


def apply_retention_model(
    layer, time, retention_model=model_endurance_retention, **retention_model_kwargs
):
    """Method to apply an retention model to devices within a memristive layer.

    Parameters
    ----------
    layer : memtorch.mn
        A memrstive layer.
    time : float
        Retention time (s).
    retention_model : function
        Retention model to use.
    retention_model_kwargs : **kwargs
        Retention model keyword arguments.

    Returns
    -------
    memtorch.mn
        The patched memristive layer.
    """
    assert retention_model in valid_retention_models, "retention_model is not valid."
    pass  # TODO
