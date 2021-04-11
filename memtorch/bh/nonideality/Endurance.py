import torch
import memtorch
import numpy as np
import math
import copy
from memtorch.bh.nonideality.endurance_retention_models.empirical_metal_oxide_RRAM import (
    model_endurance_retention,
)


def apply_endurance_model(
    layer, endurance_model=model_endurance_retention, **endurance_model_kwargs
):
    """Method to apply an endurance model to devices within a memristive layer.

    Parameters
    ----------
    layer : memtorch.mn
        A memrstive layer.
    endurance_model : function
        Endurance model to use.
    endurance_model_kwargs : **kwargs
        Endurance model keyword arguments.

    Returns
    -------
    memtorch.mn
        The patched memristive layer.
    """
    pass  # TODO
