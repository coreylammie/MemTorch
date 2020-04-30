import torch
import memtorch
import numpy as np
if 'cpu' in memtorch.__version__:
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
    device = torch.device('cpu' if 'cpu' in memtorch.__version__ else 'cuda')
    assert int(num_conductance_states) == num_conductance_states, 'num_conductance_states must be a whole number.'
    def apply_finite_conductance_states_to_crossbar(crossbar, num_conductance_states):
        crossbar_min = torch.tensor(1 / (np.vectorize(lambda x: x.r_off)(crossbar.devices))).view(-1).to(device).float()
        crossbar_max = torch.tensor(1 / (np.vectorize(lambda x: x.r_on)(crossbar.devices))).view(-1).to(device).float()
        quantization.quantize(crossbar.conductance_matrix, num_conductance_states, crossbar_min, crossbar_max)
        return crossbar

    for i in range(len(layer.crossbars)):
        layer.crossbars[i] = apply_finite_conductance_states_to_crossbar(layer.crossbars[i], num_conductance_states)

    return layer
