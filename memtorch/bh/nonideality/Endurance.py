import numpy as np
import torch

import memtorch
from memtorch.bh.crossbar.Program import naive_program
from memtorch.bh.nonideality.endurance_retention_models.empirical_metal_oxide_RRAM import (
    model_endurance_retention,
)
from memtorch.map.Parameter import naive_map

# TEMP for if __name__ == "__main__"
from memtorch.mn.Module import patch_model

valid_endurance_models = [model_endurance_retention]


def apply_endurance_model(
    layer, x, endurance_model=model_endurance_retention, **endurance_model_kwargs
):
    """Method to apply an endurance model to devices within a memristive layer.

    Parameters
    ----------
    layer : memtorch.mn
        A memrstive layer.
    x : float
        Energy (J) / SET-RESET cycles.
    endurance_model : function
        Endurance model to use.
    endurance_model_kwargs : **kwargs
        Endurance model keyword arguments.

    Returns
    -------
    memtorch.mn
        The patched memristive layer.
    """
    assert endurance_model in valid_endurance_models, "endurance_model is not valid."
    return endurance_model(layer=layer, x=x, **endurance_model_kwargs)
