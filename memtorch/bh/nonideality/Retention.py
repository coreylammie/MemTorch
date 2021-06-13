import numpy as np
import torch

import memtorch
from memtorch.bh.crossbar.Program import naive_program
from memtorch.bh.nonideality.endurance_retention_models.conductance_drift import (
    model_conductance_drift,
)
from memtorch.bh.nonideality.endurance_retention_models.empirical_metal_oxide_RRAM import (
    model_endurance_retention,
)
from memtorch.map.Parameter import naive_map

valid_retention_models = [model_endurance_retention, model_conductance_drift]


def apply_retention_model(
    layer, time, retention_model=model_conductance_drift, **retention_model_kwargs
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
    return retention_model(layer=layer, x=time, **retention_model_kwargs)


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
        non_idealities=[memtorch.bh.nonideality.NonIdeality.Retention],
        time=1e4,
        retention_model=memtorch.bh.nonideality.endurance_retention_models.model_conductance_drift,
        retention_model_kwargs={"initial_time": 0.0, "drift_coefficient": 0.1},
    )
    m_model.tune_()
