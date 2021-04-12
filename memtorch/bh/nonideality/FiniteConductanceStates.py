import copy

import numpy as np
import torch
from numpy.core.numeric import cross

import memtorch

if "cpu" in memtorch.__version__:
    import quantization
else:
    import cuda_quantization as quantization


def apply_finite_conductance_states(layer, num_conductance_states):
    """Method to model a finite number of conductance states for devices within a memristive layer.

    Parameters
    ----------
    layer : memtorch.mn
        A memrstive layer.
    num_conductance_states : int
        Number of finite conductance states to model.

    Returns
    -------
    memtorch.mn
        The patched memristive layer.
    """
    device = torch.device("cpu" if "cpu" in memtorch.__version__ else "cuda")
    assert (
        int(num_conductance_states) == num_conductance_states
    ), "num_conductance_states must be a whole number."

    def apply_finite_conductance_states_to_crossbar(crossbar, num_conductance_states):
        crossbar.update()
        conductance_matrix_ = copy.deepcopy(crossbar.conductance_matrix)
        try:
            r_on = np.nan_to_num(
                np.array(np.vectorize(lambda x: x.r_on)(crossbar.devices)),
                copy=False,
                nan=crossbar.r_on_mean,
                posinf=crossbar.r_on_mean,
                neginf=crossbar.r_on_mean,
            )
            r_on[r_on == 0] = crossbar.r_on_mean
            r_off = np.nan_to_num(
                np.array(np.vectorize(lambda x: x.r_off)(crossbar.devices)),
                copy=False,
                nan=crossbar.r_off_mean,
                posinf=crossbar.r_off_mean,
                neginf=crossbar.r_off_mean,
            )
            r_off[r_off == 0] = crossbar.r_off_mean
            if np.unique(r_on).size == 1:
                r_on = torch.ones(
                    crossbar.conductance_matrix.shape, device=device
                ).float() * float(crossbar.r_on_mean)
            else:
                r_on = torch.from_numpy(r_on).float().cuda()

            if np.unique(r_off).size == 1:
                r_off = torch.ones(
                    crossbar.conductance_matrix.shape, device=device
                ).float() * float(crossbar.r_off_mean)
            else:
                r_off = torch.from_numpy(r_off).float().cuda()

            conductance_matrix_shape = crossbar.conductance_matrix.shape
            conductance_matrix = crossbar.conductance_matrix.view(-1)
            quantization.quantize(
                conductance_matrix,
                num_conductance_states,
                1 / r_off.view(-1),
                1 / r_on.view(-1),
            )
            conductance_matrix = conductance_matrix.view(conductance_matrix_shape)
            conductance_matrix[0]
            crossbar.conductance_matrix = conductance_matrix.view(
                conductance_matrix_shape
            ).float()
        except:
            crossbar.conductance_matrix = conductance_matrix_.float()

        return crossbar

    for i in range(len(layer.crossbars)):
        layer.crossbars[i] = apply_finite_conductance_states_to_crossbar(
            layer.crossbars[i], num_conductance_states
        )

    return layer
