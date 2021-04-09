import torch
import memtorch
import numpy as np
import math
import copy
import matplotlib.pyplot as plt


def apply_endurance_model(layer, endurance_model, endurance_model_kwargs):
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
    pass # TODO

