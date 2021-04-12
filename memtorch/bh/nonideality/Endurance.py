import torch
import memtorch
import numpy as np
import math
import copy
from memtorch.bh.nonideality.endurance_retention_models.empirical_metal_oxide_RRAM import (
    model_endurance_retention,
)

# TEMP for if __name__ == "__main__"
from memtorch.mn.Module import patch_model
from memtorch.map.Parameter import naive_map
from memtorch.bh.crossbar.Program import naive_program


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


if __name__ == "__main__":

    class MLP(torch.nn.Module):
        def __init__(self, n_inputs):
            super(MLP, self).__init__()
            self.layer = torch.nn.Linear(n_inputs, 10)
            self.activation = torch.nn.ReLU()

        def forward(self, x):
            x = self.layer(x)
            x = self.activation(x)
            return x

    model = MLP(n_inputs=100)
    reference_memristor = memtorch.bh.memristor.VTEAM
    m_model = patch_model(
        model,
        memristor_model=reference_memristor,
        memristor_model_params={},
        module_parameters_to_patch=[torch.nn.Linear],
        mapping_routine=naive_map,
        transistor=True,
        programming_routine=None,
        scheme=memtorch.bh.Scheme.DoubleColumn,
        tile_shape=(8, 8),
        max_input_voltage=0.3,
        ADC_resolution=int(8),
        ADC_overflow_rate=0.0,
        quant_method="linear",
    )
    m_model.tune_()
    p_model = memtorch.bh.nonideality.apply_nonidealities(
        m_model,
        non_idealities=[memtorch.bh.nonideality.NonIdeality.Endurance],
        x=1e4,
        endurance_model=memtorch.bh.nonideality.endurance_retention_models.model_endurance_retention,
        endurance_model_kwargs={
            "operation_mode": memtorch.bh.nonideality.endurance_retention_models.OperationMode.sudden,
            "p_lrs": [0, 0, 0],
            "stable_resistance_lrs": 100,
            "p_hrs": [0, 0, 0],
            "stable_resistance_hrs": 1000,
            "cell_size": None,
            "tempurature": None,
        },
    )
    m_model.tune_()
