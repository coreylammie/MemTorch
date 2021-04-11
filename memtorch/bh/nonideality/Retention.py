import torch
import memtorch
import numpy as np
import math
import copy
import matplotlib.pyplot as plt


def apply_retention_model(layer, retention_model, retention_model_kwargs):
    """Method to apply an retention model to devices within a memristive layer.

    Parameters
    ----------
    layer : memtorch.mn
        A memrstive layer.
    retention_model : function
        Retention model to use.
    retention_model_kwargs : **kwargs
        Retention model keyword arguments.

    Returns
    -------
    memtorch.mn
        The patched memristive layer.
    """
    pass  # TODO
